"""rag_pipeline.py"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import cache as cache_module
from jd_parser import parse_job_description
from resume_analyzer import chunk_resume_by_sections, parse_resume
from scorer import score_resume, WEIGHTS
from utils import Timer, get_logger, hash_text

logger = get_logger()

EMBEDDING_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-latest")

MMR_K = 6
MMR_FETCH_K = 15
MMR_LAMBDA = 0.5
MAX_WORKERS = int(os.getenv("RESUME_SCREENER_MAX_WORKERS", "8"))

EMPTY_CATEGORY_SCORES = {"skills": 0, "projects": 0, "experience": 0, "education": 0, "certifications": 0}

_CATEGORY_WEIGHTS = {
    "skills": WEIGHTS["required_skills"] + WEIGHTS["preferred_skills"],
    "projects": WEIGHTS["projects"],
    "experience": WEIGHTS["experience"],
    "education": WEIGHTS["education"],
    "certifications": WEIGHTS["certifications"],
}


def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)


def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.1)


def _build_score_breakdown(category_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Attach weight and weighted-point-contribution context to each category score."""
    breakdown = {}
    for category, score in category_scores.items():
        weight = _CATEGORY_WEIGHTS.get(category, 0)
        breakdown[category] = {
            "score": score,
            "weight_percent": weight,
            "weighted_contribution": round(score * weight / 100, 2),
        }
    return breakdown


def _empty_result(candidate_name: str, message: str, jd_struct: Dict) -> Dict:
    """Full-schema placeholder result for resumes that fail before scoring."""
    return {
        "candidate_name": candidate_name,
        "overall_score": 0,
        "category_scores": dict(EMPTY_CATEGORY_SCORES),
        "score_breakdown": _build_score_breakdown(EMPTY_CATEGORY_SCORES),
        "matched_skills": [],
        "missing_skills": jd_struct.get("required_skills", []) if jd_struct else [],
        "matched_certifications": [],
        "missing_certifications": [],
        "relevant_projects": [],
        "keyword_evidence": [],
        "strengths": [],
        "weaknesses": [],
        "recommendation": "Do Not Shortlist",
        "summary": message,
        "extracted_resume_facts": {},
        "evidence_chunks": [],
    }


def _parse_jd_cached(job_description: str, llm) -> Dict:
    """Parse the JD once, reusing a cached structured result for identical JD text."""
    jd_hash = hash_text(job_description)
    cached = cache_module.get_jd(jd_hash)
    if cached is not None:
        logger.info(f"JD cache hit ({jd_hash[:12]}...)")
        return cached

    with Timer(logger, "Gemini JD extraction"):
        jd_struct = parse_job_description(job_description, llm)

    if "_error" in jd_struct:
        logger.warning(f"JD extraction failed: {jd_struct['_error']}")
        return jd_struct

    cache_module.save_jd(jd_hash, jd_struct)
    return jd_struct


def _get_or_embed_jd(jd_hash: str, job_description: str, embeddings) -> List[float]:
    """Embed the JD once; reuse a cached vector for identical JD text on future runs."""
    cached = cache_module.get_embeddings(jd_hash, namespace="jd_embeddings")
    if cached is not None and cached.get("texts") == [job_description]:
        logger.info(f"JD embedding cache hit ({jd_hash[:12]}...)")
        return cached["vectors"][0]

    vector = embeddings.embed_query(job_description)
    cache_module.save_embeddings(jd_hash, [job_description], [vector], namespace="jd_embeddings")
    return vector


def _get_or_embed_resume_chunks(
    resume_hash: str, texts: List[str], embeddings
) -> Optional[List[List[float]]]:
    """Embed resume chunks once per resume; reuse cached vectors across screening runs."""
    if not texts:
        return None

    cached = cache_module.get_embeddings(resume_hash, namespace="resume_embeddings")
    if cached is not None and cached.get("texts") == texts:
        logger.info(f"Resume chunk embedding cache hit ({resume_hash[:12]}...)")
        return cached["vectors"]

    try:
        with Timer(logger, f"Embedding {len(texts)} resume chunk(s) ({resume_hash[:12]}...)"):
            vectors = embeddings.embed_documents(texts)
    except Exception as e:
        logger.warning(f"Resume chunk embedding failed for {resume_hash[:12]}...: {e}")
        return None

    cache_module.save_embeddings(resume_hash, texts, vectors, namespace="resume_embeddings")
    return vectors


def _load_or_build_vectorstore(
    resume_hash: str, section_chunks: List[Tuple[str, str]], embeddings
) -> Optional[FAISS]:
    cached_dir = cache_module.get_vectorstore_dir(resume_hash)
    if cached_dir is not None:
        try:
            with Timer(logger, f"Loading cached FAISS index ({resume_hash[:12]}...)"):
                vectorstore = FAISS.load_local(
                    str(cached_dir), embeddings, allow_dangerous_deserialization=True
                )
            return vectorstore
        except Exception as e:
            logger.warning(
                f"Cached FAISS index for {resume_hash[:12]}... failed to load, rebuilding: {e}"
            )

    texts = [t for _, t in section_chunks]
    sections = [s for s, _ in section_chunks]
    vectors = _get_or_embed_resume_chunks(resume_hash, texts, embeddings)
    if not vectors:
        return None

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embeddings,
        metadatas=[{"section": s} for s in sections],
    )

    try:
        save_dir = cache_module.vectorstore_dir_for_saving(resume_hash)
        vectorstore.save_local(str(save_dir))
        cache_module.save_vectorstore_marker(resume_hash)
    except Exception as e:
        logger.warning(f"Could not persist FAISS index for {resume_hash[:12]}...: {e}")

    return vectorstore


def _build_evidence_query(report: Dict) -> str:
    parts = []
    if report.get("matched_skills"):
        parts.append("Skills: " + ", ".join(report["matched_skills"]))
    if report.get("relevant_projects"):
        parts.append("Projects: " + ", ".join(report["relevant_projects"]))
    if report.get("matched_certifications"):
        parts.append("Certifications: " + ", ".join(report["matched_certifications"]))
    return ". ".join(parts) or "skills experience education certifications"


def _retrieve_evidence(resume_hash: str, resume_text: str, report: Dict, embeddings) -> List[Dict[str, str]]:
    """
    Post-scoring MMR retrieval limited to evidence for what was matched.
    Never used for structured extraction — extraction already happened on
    the complete resume text in resume_analyzer.parse_resume.
    """
    section_chunks = chunk_resume_by_sections(resume_text)

    vectorstore = _load_or_build_vectorstore(resume_hash, section_chunks, embeddings)
    if vectorstore is None:
        return []

    try:
        query_text = _build_evidence_query(report)
        query_vector = embeddings.embed_query(query_text)
        docs = vectorstore.max_marginal_relevance_search_by_vector(
            embedding=query_vector, k=MMR_K, fetch_k=MMR_FETCH_K, lambda_mult=MMR_LAMBDA,
        )
        return [{"section": d.metadata.get("section", "general"), "text": d.page_content} for d in docs]
    except Exception as e:
        logger.warning(f"Evidence retrieval failed for {resume_hash[:12]}...: {e}")
        return []


def _process_single_resume(
    filename: str, resume_text: str, jd_struct: Dict, llm, embeddings
) -> Dict:
    """Full per-resume pipeline: parse complete resume -> score -> retrieve evidence."""
    if not resume_text or not resume_text.strip():
        return _empty_result(filename, "No extractable text found in this file.", jd_struct)

    resume_hash = hash_text(resume_text)

    try:
        with Timer(logger, f"Resume parse: {filename}"):
            resume_struct = parse_resume(filename, resume_text, llm)
    except Exception as e:
        logger.error(f"Unhandled error parsing resume {filename}: {e}")
        return _empty_result(filename, f"⚠️ Resume parsing failed: {e}", jd_struct)

    if "_error" in resume_struct:
        return _empty_result(filename, f"⚠️ {resume_struct['_error']}", jd_struct)

    try:
        with Timer(logger, f"ATS scoring: {filename}"):
            report = score_resume(filename, jd_struct, resume_struct, embeddings)
    except Exception as e:
        logger.error(f"Unhandled scoring error for {filename}: {e}")
        return _empty_result(filename, f"⚠️ Scoring failed: {e}", jd_struct)

    report["score_breakdown"] = _build_score_breakdown(report["category_scores"])
    report["extracted_resume_facts"] = resume_struct

    try:
        with Timer(logger, f"Evidence retrieval: {filename}"):
            report["evidence_chunks"] = _retrieve_evidence(resume_hash, resume_text, report, embeddings)
    except Exception as e:
        logger.warning(f"Evidence retrieval step failed for {filename}: {e}")
        report["evidence_chunks"] = []

    return report


def rank_resumes(job_description: str, resumes: List[Tuple[str, str]], api_key: str) -> Dict:
    """
    Screen a batch of resumes against a job description.
    resumes: list of (filename, resume_text) tuples.
    Returns {"jd_struct": ..., "results": [...], "jd_error": Optional[str]}.
    Results are sorted by overall_score descending, with ties broken by
    original input order — deterministic regardless of which worker
    thread happens to finish first.
    """
    embeddings = get_embeddings(api_key)  # single instance, reused for every resume below
    llm = get_llm(api_key)                # single instance, reused for every resume below

    with Timer(logger, "JD parsing"):
        jd_struct = _parse_jd_cached(job_description, llm)

    if "_error" in jd_struct:
        return {"jd_struct": jd_struct, "results": [], "jd_error": jd_struct["_error"]}

    jd_hash = hash_text(job_description)
    try:
        _get_or_embed_jd(jd_hash, job_description, embeddings)
    except Exception as e:
        logger.warning(f"JD embedding failed (non-fatal, evidence retrieval may degrade): {e}")

    input_order = {filename: i for i, (filename, _) in enumerate(resumes)}
    results: List[Dict] = []
    worker_count = max(1, min(MAX_WORKERS, len(resumes) or 1))

    with Timer(logger, f"Screening {len(resumes)} resume(s) with {worker_count} worker(s)"):
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_process_single_resume, filename, text, jd_struct, llm, embeddings): filename
                for filename, text in resumes
            }
            for future in as_completed(future_map):
                filename = future_map[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Unhandled failure processing {filename}: {e}")
                    results.append(_empty_result(filename, f"⚠️ Unexpected error: {e}", jd_struct))

    results.sort(key=lambda r: (-r.get("overall_score", 0), input_order.get(r["candidate_name"], 0)))
    return {"jd_struct": jd_struct, "results": results, "jd_error": None}