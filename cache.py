"""
cache.py
---------
Persistent, thread-safe caching for everything expensive in the pipeline:
parsed resume JSON, parsed JD JSON, resume-chunk embeddings, and skill
embeddings — plus (when practical) full FAISS vectorstores.

Design summary (see chat for full rationale):
- Disk-backed JSON files, keyed by SHA-256 content hash (utils.hash_bytes /
  utils.hash_text) — never by filename. This is what makes "resume renamed
  -> still a hit" and "resume edited -> automatic miss" both true without
  any explicit invalidation code.
- A small CacheBackend interface separates *what* gets cached (this file)
  from *where* it's stored, so swapping disk storage for Redis later means
  writing one new backend class, not touching the rest of the app.
- Every entry carries a schema_version + created_at, so a prompt/schema
  change can invalidate everything at once (bump CACHE_SCHEMA_VERSION),
  and TTLs catch staleness even when content hashes haven't changed.
- Only the in-memory hit/miss counters need a lock; concurrent writes to
  different cache keys are naturally safe since they're different files.
"""

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from utils import get_logger

logger = get_logger()

# Bump this if prompts.py or a struct schema changes materially enough
# that old cached extractions should no longer be trusted.
CACHE_SCHEMA_VERSION = 1

DEFAULT_CACHE_DIR = Path(os.getenv("RESUME_SCREENER_CACHE_DIR", Path(__file__).resolve().parent / ".cache"))

SECONDS_PER_DAY = 60 * 60 * 24
DEFAULT_TTLS = {
    "resume": 90 * SECONDS_PER_DAY,
    "jd": 7 * SECONDS_PER_DAY,
    "resume_embeddings": 90 * SECONDS_PER_DAY,
    "skill_embeddings": None,  # normalized skill -> vector essentially never goes stale
    "vectorstore": 90 * SECONDS_PER_DAY,
}

NAMESPACES = ["resume", "jd", "resume_embeddings", "skill_embeddings", "vectorstore"]


# ---------------------------------------------------------------------
# Backend interface — swap DiskCacheBackend for RedisCacheBackend (or
# anything else implementing this interface) without touching the rest
# of the app.
# ---------------------------------------------------------------------

class CacheBackend(Protocol):
    def read(self, namespace: str, key: str) -> Optional[dict]: ...
    def write(self, namespace: str, key: str, payload: dict) -> None: ...
    def delete(self, namespace: str, key: str) -> None: ...
    def exists(self, namespace: str, key: str) -> bool: ...
    def list_keys(self, namespace: str) -> list: ...
    def size_bytes(self, namespace: str) -> int: ...
    def path_for(self, namespace: str, key: str) -> Path:
        """Only meaningful for file-based backends (e.g. FAISS save_local target)."""
        ...


class DiskCacheBackend:
    """Default backend: one JSON file per (namespace, key) under root/namespace/key.json."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        for ns in NAMESPACES:
            (self.root / ns).mkdir(parents=True, exist_ok=True)

    def _file_path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.json"

    def path_for(self, namespace: str, key: str) -> Path:
        """Directory reserved for non-JSON entries (e.g. a FAISS index folder)."""
        directory = self.root / namespace / key
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def read(self, namespace: str, key: str) -> Optional[dict]:
        path = self._file_path(namespace, key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cache read failed for {namespace}/{key}: {e} — treating as miss")
            return None

    def write(self, namespace: str, key: str, payload: dict) -> None:
        path = self._file_path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)  # covers namespaces (e.g. "_meta") not in NAMESPACES
        # Unique temp name per call: multiple threads can legitimately write
        # the SAME key concurrently (e.g. the shared stats file on every
        # save_*), so a fixed ".tmp" name would let one thread's rename
        # race another's. os.replace to the shared final path is still
        # atomic — the last writer simply wins.
        tmp_path = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)  # atomic on POSIX and Windows

    def delete(self, namespace: str, key: str) -> None:
        path = self._file_path(namespace, key)
        if path.exists():
            path.unlink()
        # Also remove a directory-style entry (e.g. a cached FAISS index), if any.
        directory = self.root / namespace / key
        if directory.exists() and directory.is_dir():
            import shutil
            shutil.rmtree(directory, ignore_errors=True)

    def exists(self, namespace: str, key: str) -> bool:
        return self._file_path(namespace, key).exists() or (self.root / namespace / key).exists()

    def list_keys(self, namespace: str) -> list:
        ns_dir = self.root / namespace
        if not ns_dir.exists():
            return []
        json_keys = [p.stem for p in ns_dir.glob("*.json")]
        dir_keys = [p.name for p in ns_dir.iterdir() if p.is_dir()]
        return list(set(json_keys + dir_keys))

    def size_bytes(self, namespace: str) -> int:
        ns_dir = self.root / namespace
        if not ns_dir.exists():
            return 0
        total = 0
        for p in ns_dir.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total


# ---------------------------------------------------------------------
# Domain-facing cache
# ---------------------------------------------------------------------

@dataclass
class _NamespaceStats:
    hits: int = 0
    misses: int = 0
    saves: int = 0
    last_updated: Optional[float] = None


class ResumeScreenerCache:
    """
    The API every other module talks to. Wraps a CacheBackend (disk by
    default) with TTL/schema-version checks and hit/miss bookkeeping.
    """

    def __init__(self, backend: Optional[CacheBackend] = None, ttls: Optional[dict] = None):
        self.backend = backend or DiskCacheBackend(DEFAULT_CACHE_DIR)
        self.ttls = ttls or DEFAULT_TTLS
        self._stats_lock = threading.RLock()
        self._stats = {ns: _NamespaceStats() for ns in NAMESPACES}
        self._load_persisted_stats()

    # -- internal: entry wrapping / TTL + schema checks -----------------

    def _wrap(self, data) -> dict:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "created_at": time.time(),
            "data": data,
        }

    def _unwrap_if_valid(self, namespace: str, entry: Optional[dict]):
        if entry is None:
            return None
        if entry.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        ttl = self.ttls.get(namespace)
        if ttl is not None and (time.time() - entry.get("created_at", 0)) > ttl:
            return None
        return entry.get("data")

    def _record(self, namespace: str, hit: bool) -> None:
        with self._stats_lock:
            stats = self._stats[namespace]
            if hit:
                stats.hits += 1
            else:
                stats.misses += 1

    def _record_save(self, namespace: str) -> None:
        with self._stats_lock:
            stats = self._stats[namespace]
            stats.saves += 1
            stats.last_updated = time.time()
        self._persist_stats()

    # -- internal: stats persistence (cheap — only on save/cache_stats) --

    _STATS_NAMESPACE = "_meta"
    _STATS_KEY = "stats"

    def _load_persisted_stats(self) -> None:
        try:
            raw = self.backend.read(self._STATS_NAMESPACE, self._STATS_KEY)
        except Exception:
            raw = None
        if not raw:
            return
        data = raw.get("data", {})
        with self._stats_lock:
            for ns, values in data.items():
                if ns in self._stats:
                    self._stats[ns] = _NamespaceStats(**values)

    def _persist_stats(self) -> None:
        with self._stats_lock:
            snapshot = {ns: vars(s) for ns, s in self._stats.items()}
        try:
            self.backend.write(self._STATS_NAMESPACE, self._STATS_KEY, self._wrap(snapshot))
        except Exception as e:
            logger.warning(f"Could not persist cache stats: {e}")

    # -- Resume JSON ------------------------------------------------------

    def get_resume(self, resume_hash: str) -> Optional[dict]:
        entry = self.backend.read("resume", resume_hash)
        data = self._unwrap_if_valid("resume", entry)
        self._record("resume", hit=data is not None)
        return data

    def save_resume(self, resume_hash: str, resume_struct: dict) -> None:
        self.backend.write("resume", resume_hash, self._wrap(resume_struct))
        self._record_save("resume")

    # -- Job Description JSON ---------------------------------------------

    def get_jd(self, jd_hash: str) -> Optional[dict]:
        entry = self.backend.read("jd", jd_hash)
        data = self._unwrap_if_valid("jd", entry)
        self._record("jd", hit=data is not None)
        return data

    def save_jd(self, jd_hash: str, jd_struct: dict) -> None:
        self.backend.write("jd", jd_hash, self._wrap(jd_struct))
        self._record_save("jd")

    # -- Embeddings (generic; used for resume-chunk embeddings) -----------

    def get_embeddings(self, key_hash: str, namespace: str = "resume_embeddings") -> Optional[dict]:
        """Returns {"texts": [...], "vectors": [...]} or None on a miss."""
        entry = self.backend.read(namespace, key_hash)
        data = self._unwrap_if_valid(namespace, entry)
        self._record(namespace, hit=data is not None)
        return data

    def save_embeddings(self, key_hash: str, texts: list, vectors: list,
                         namespace: str = "resume_embeddings") -> None:
        self.backend.write(namespace, key_hash, self._wrap({"texts": texts, "vectors": vectors}))
        self._record_save(namespace)

    # -- Skill embeddings (one vector per normalized skill string) --------

    def get_skill_embedding(self, skill_hash: str) -> Optional[list]:
        entry = self.backend.read("skill_embeddings", skill_hash)
        data = self._unwrap_if_valid("skill_embeddings", entry)
        self._record("skill_embeddings", hit=data is not None)
        return data.get("vector") if data else None

    def save_skill_embedding(self, skill_hash: str, vector: list) -> None:
        self.backend.write("skill_embeddings", skill_hash, self._wrap({"vector": vector}))
        self._record_save("skill_embeddings")

    # -- FAISS vectorstore (optional; cached "when practical") ------------

    def get_vectorstore_dir(self, resume_hash: str):
        """
        Returns the directory path to load an existing cached vectorstore
        from (via FAISS.load_local(path, embeddings)), or None if there's
        no valid, non-expired cache entry for this resume.
        """
        marker = self.backend.read("vectorstore", f"{resume_hash}.marker")
        data = self._unwrap_if_valid("vectorstore", marker)
        self._record("vectorstore", hit=data is not None)
        if data is None:
            return None
        return self.backend.path_for("vectorstore", resume_hash)

    def save_vectorstore_marker(self, resume_hash: str) -> None:
        """
        Call this AFTER vectorstore.save_local(cache.path_for(...)) succeeds,
        so the marker's created_at reflects a genuinely completed save.
        """
        self.backend.write("vectorstore", f"{resume_hash}.marker", self._wrap({"resume_hash": resume_hash}))
        self._record_save("vectorstore")

    def vectorstore_dir_for_saving(self, resume_hash: str) -> Path:
        return self.backend.path_for("vectorstore", resume_hash)

    # -- Invalidation ------------------------------------------------------

    def invalidate_resume(self, resume_hash: str) -> None:
        """Removes this resume's struct, embeddings, and cached vectorstore together."""
        self.backend.delete("resume", resume_hash)
        self.backend.delete("resume_embeddings", resume_hash)
        self.backend.delete("vectorstore", f"{resume_hash}.marker")
        self.backend.delete("vectorstore", resume_hash)
        logger.info(f"Invalidated all cache entries for resume {resume_hash[:12]}...")

    def invalidate_jd(self, jd_hash: str) -> None:
        """Provided for symmetry/manual cleanup — a changed JD already gets a new
        hash automatically, so this never needs to run for correctness, only
        for reclaiming disk space on demand."""
        self.backend.delete("jd", jd_hash)
        logger.info(f"Invalidated JD cache entry {jd_hash[:12]}...")

    def clear_cache(self, namespace: Optional[str] = None) -> None:
        """Clear one namespace, or everything if namespace is None."""
        targets = [namespace] if namespace else NAMESPACES
        for ns in targets:
            for key in self.backend.list_keys(ns):
                self.backend.delete(ns, key)
            with self._stats_lock:
                self._stats[ns] = _NamespaceStats()
        self._persist_stats()
        logger.info(f"Cleared cache namespace(s): {targets}")

    # -- Stats --------------------------------------------------------------

    def cache_stats(self) -> dict:
        """
        Returns hits/misses/saves per namespace, total disk size, last-updated
        timestamps, and an estimated count of Gemini API calls saved by cache
        hits (one saved call per resume/JD hit; embedding hits are a coarser
        estimate since a single embedding call can cover many chunks/skills).
        """
        with self._stats_lock:
            snapshot = {ns: dict(vars(s)) for ns, s in self._stats.items()}

        for ns in NAMESPACES:
            snapshot[ns]["size_bytes"] = self.backend.size_bytes(ns)
            snapshot[ns]["entry_count"] = len(self.backend.list_keys(ns))

        saved_calls = snapshot["resume"]["hits"] + snapshot["jd"]["hits"]
        saved_embedding_calls = snapshot["resume_embeddings"]["hits"] + snapshot["skill_embeddings"]["hits"]

        self._persist_stats()

        return {
            "namespaces": snapshot,
            "total_size_bytes": sum(v["size_bytes"] for v in snapshot.values()),
            "estimated_chat_calls_saved": saved_calls,
            "estimated_embedding_calls_saved": saved_embedding_calls,
        }


# ---------------------------------------------------------------------
# Module-level singleton + thin functional API
# (so callers can `from cache import get_resume, save_resume, ...`)
# ---------------------------------------------------------------------

_default_cache = ResumeScreenerCache()


def get_resume(resume_hash: str) -> Optional[dict]:
    return _default_cache.get_resume(resume_hash)


def save_resume(resume_hash: str, resume_struct: dict) -> None:
    _default_cache.save_resume(resume_hash, resume_struct)


def get_jd(jd_hash: str) -> Optional[dict]:
    return _default_cache.get_jd(jd_hash)


def save_jd(jd_hash: str, jd_struct: dict) -> None:
    _default_cache.save_jd(jd_hash, jd_struct)


def get_embeddings(key_hash: str, namespace: str = "resume_embeddings") -> Optional[dict]:
    return _default_cache.get_embeddings(key_hash, namespace)


def save_embeddings(key_hash: str, texts: list, vectors: list, namespace: str = "resume_embeddings") -> None:
    _default_cache.save_embeddings(key_hash, texts, vectors, namespace)


def get_vectorstore_dir(resume_hash: str):
    """Directory of a cached FAISS index for this resume hash, or None if absent/expired."""
    return _default_cache.get_vectorstore_dir(resume_hash)


def vectorstore_dir_for_saving(resume_hash: str) -> Path:
    """Directory to pass to vectorstore.save_local(...) for this resume hash."""
    return _default_cache.vectorstore_dir_for_saving(resume_hash)


def save_vectorstore_marker(resume_hash: str) -> None:
    """Call after vectorstore.save_local(...) succeeds to record the cache entry."""
    _default_cache.save_vectorstore_marker(resume_hash)


def get_skill_embedding(skill_hash: str) -> Optional[list]:
    return _default_cache.get_skill_embedding(skill_hash)


def save_skill_embedding(skill_hash: str, vector: list) -> None:
    _default_cache.save_skill_embedding(skill_hash, vector)


def invalidate_resume(resume_hash: str) -> None:
    _default_cache.invalidate_resume(resume_hash)


def invalidate_jd(jd_hash: str) -> None:
    _default_cache.invalidate_jd(jd_hash)


def clear_cache(namespace: Optional[str] = None) -> None:
    _default_cache.clear_cache(namespace)


def cache_stats() -> dict:
    return _default_cache.cache_stats()