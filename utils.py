"""utils.py"""

import hashlib
import json
import logging
import re
import time
from typing import Optional

_LOGGER_NAME = "resume_screener"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a configured singleton logger with a single console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class Timer:
    """Context manager that logs how long a pipeline stage took."""

    def __init__(self, logger: logging.Logger, label: str):
        self.logger = logger
        self.label = label
        self.start: Optional[float] = None
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.elapsed_seconds = time.perf_counter() - (self.start or time.perf_counter())
        if exc_type is None:
            self.logger.info(f"{self.label} — done in {self.elapsed_seconds:.2f}s")
        else:
            self.logger.warning(f"{self.label} — failed after {self.elapsed_seconds:.2f}s: {exc_val}")
        return False


def hash_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of normalized (stripped, lowercased) text."""
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_text_content(content) -> str:
    """Normalize a LangChain chat response's .content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", "")))
        return "".join(parts)
    return str(content) if content is not None else ""


def safe_parse_json(raw_text: str) -> dict:
    """Parse a JSON object out of a model response, tolerating markdown fences or stray text."""
    cleaned = (raw_text or "").strip()
    cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model response. Started with: {cleaned[:200]!r}")


def call_gemini_json(llm, prompt: str, fallback: dict) -> dict:
    """Call the LLM expecting JSON-only output; return fallback + '_error' on any failure."""
    logger = get_logger()
    try:
        response = llm.invoke(prompt)
        raw_text = extract_text_content(response.content)
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
        result = dict(fallback)
        result["_error"] = f"Gemini API call failed: {e}"
        return result

    try:
        parsed = safe_parse_json(raw_text)
    except Exception as e:
        logger.warning(f"Failed to parse Gemini JSON response: {e}")
        result = dict(fallback)
        result["_error"] = f"{e}"
        return result

    return parsed