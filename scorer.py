"""scorer.py"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from skill_normalizer import (
    EDUCATION_LEVELS,
    match_skills,
    normalize_education,
    embed_with_cache,
)
from utils import get_logger, Timer

logger = get_logger()

WEIGHTS: Dict[str, float] = {
    "required_skills": 50,
    "preferred_skills": 5,
    "projects": 15,
    "experience": 15,
    "education": 10,
    "certifications": 5,
}

PROJECT_RELEVANCE_THRESHOLD = 0.80


def _ratio_score(matched_count: int, total_count: int) -> float:
    """Fraction matched as a 0-100 score. No requirement (total=0) yields full score."""
    if total_count == 0:
        return 100.0
    return min(100.0, (matched_count / total_count) * 100.0)


def _score_experience(min_years: Optional[float], resume_years: Optional[float]) -> float:
    """Score experience match; no JD requirement yields full score."""
    if min_years is None or min_years == 0:
        return 100.0
    if resume_years is None:
        return 0.0
    if resume_years >= min_years:
        return 100.0
    return max(0.0, (resume_years / min_years) * 100.0)


def _score_education(required_level: Optional[str], resume_level: Optional[str]) -> float:
    """Score education match using normalized levels; no requirement yields full score."""
    norm_required = normalize_education(required_level)
    norm_resume = normalize_education(resume_level)

    if not norm_required or norm_required not in EDUCATION_LEVELS:
        return 100.0
    if not norm_resume or norm_resume not in EDUCATION_LEVELS:
        return 40.0

    required_rank = EDUCATION_LEVELS[norm_required]
    resume_rank = EDUCATION_LEVELS[norm_resume]
    if resume_rank >= required_rank:
        return 100.0
    gap = required_rank - resume_rank
    return max(0.0, 100.0 - gap * 35.0)


def _project_text(project: Dict) -> str:
    name = project.get("name") or ""
    description = project.get("description") or ""
    technologies = project.get("technologies_used") or []
    return f"{name} {description} {' '.join(technologies)}".strip()


def _score_projects_semantic(
    jd_relevant_terms: List[str],
    projects: List[Dict],
    embeddings,
) -> Tuple[float, List[str]]:
    """
    Score project relevance via semantic similarity between JD required/
    preferred skills and each project's combined text (name + description
    + technologies_used), instead of keyword substring matching.
    Returns (score, relevant_project_names).
    """
    if not projects:
        return 50.0, []

    terms = [t for t in jd_relevant_terms if t]
    if not terms:
        return 70.0, []

    project_texts = [_project_text(p) for p in projects]
    project_texts = [t if t else "unspecified project" for t in project_texts]

    try:
        with Timer(logger, f"Semantic project scoring ({len(projects)} project(s))"):
            combined = terms + project_texts
            vectors = embed_with_cache(combined, embeddings)
            term_vecs = vectors[: len(terms)]
            project_vecs = vectors[len(terms):]

            term_norm = term_vecs / (np.linalg.norm(term_vecs, axis=1, keepdims=True) + 1e-8)
            project_norm = project_vecs / (np.linalg.norm(project_vecs, axis=1, keepdims=True) + 1e-8)
            sim_matrix = project_norm @ term_norm.T
            best_sim_per_project = sim_matrix.max(axis=1)
    except Exception as e:
        logger.warning(f"Semantic project scoring failed, falling back to neutral score: {e}")
        return 60.0, []

    relevant_names = [
        (projects[i].get("name") or f"Project {i + 1}")
        for i, sim in enumerate(best_sim_per_project)
        if sim >= PROJECT_RELEVANCE_THRESHOLD
    ]

    avg_sim = float(np.mean(best_sim_per_project))
    score = min(100.0, max(0.0, avg_sim * 100.0))
    return score, relevant_names

def _score_certifications(
    jd_certifications: List[str],
    resume_certifications: List[str],
    embeddings,
) -> Tuple[float, List[str], List[str]]:
    """Score certification match; returns (score, matched, missing)."""
    if not jd_certifications:
        bonus = min(100.0, len(resume_certifications or []) * 20.0)
        return bonus, list(resume_certifications or []), []

    match_result = match_skills(jd_certifications, resume_certifications or [], embeddings)
    score = _ratio_score(len(match_result["matched"]), len(jd_certifications))
    return score, match_result["matched"], match_result["missing"]


def _keyword_evidence(jd_keywords: List[str], resume_terms: List[str]) -> List[str]:
    """
    Additional explainability signal only — overlap between JD keywords and
    resume terms. Never contributes to the numeric score.
    """
    if not jd_keywords:
        return []
    try:
        match_result = match_skills(jd_keywords, resume_terms, embeddings=None) if False else None
    except Exception:
        match_result = None

    resume_lower = {t.lower() for t in resume_terms if t}
    return [kw for kw in jd_keywords if kw and kw.lower() in resume_lower]


def _recommendation(overall_score: float) -> str:
    if overall_score >= 75:
        return "Shortlist"
    if overall_score >= 50:
        return "Consider"
    return "Do Not Shortlist"


def _build_summary(
    candidate_name: str,
    overall_score: float,
    matched_required: List[str],
    missing_required: List[str],
    recommendation: str,
) -> str:
    matched_preview = ", ".join(matched_required[:3]) if matched_required else "no directly matched required skills"
    missing_preview = ", ".join(missing_required[:3]) if missing_required else "no major gaps"
    return (
        f"{candidate_name} scored {overall_score}/100 against this role. "
        f"Strongest overlap: {matched_preview}. "
        f"Key gaps: {missing_preview}. "
        f"Recommendation: {recommendation}."
    )


def score_resume(candidate_name: str, jd_struct: Dict, resume_struct: Dict, embeddings) -> Dict:
    """
    Compute the full deterministic ATS report for one candidate from
    already-extracted structured JD and resume facts. No Gemini calls
    occur in this function.
    """
    try:
        resume_skills_all = list(
            set(
                (resume_struct.get("skills") or [])
                + (resume_struct.get("tools") or [])
                + (resume_struct.get("programming_languages") or [])
                + (resume_struct.get("frameworks") or [])
                + (resume_struct.get("databases") or [])
                + (resume_struct.get("cloud_platforms") or [])
                + (resume_struct.get("ats_keywords") or [])
            )
        )

        required_match = match_skills(jd_struct.get("required_skills") or [], resume_skills_all, embeddings)
        preferred_match = match_skills(jd_struct.get("preferred_skills") or [], resume_skills_all, embeddings)

        required_score = _ratio_score(len(required_match["matched"]), len(jd_struct.get("required_skills") or []))
        preferred_score = _ratio_score(len(preferred_match["matched"]), len(jd_struct.get("preferred_skills") or []))

        total_skill_weight = WEIGHTS["required_skills"] + WEIGHTS["preferred_skills"]
        skills_category_score = round(
            (required_score * WEIGHTS["required_skills"] + preferred_score * WEIGHTS["preferred_skills"])
            / total_skill_weight
        )

        relevant_terms = (jd_struct.get("required_skills") or []) + (jd_struct.get("preferred_skills") or [])
        projects_score, relevant_projects = _score_projects_semantic(
            relevant_terms, resume_struct.get("projects") or [], embeddings
        )

        experience_score = _score_experience(
            jd_struct.get("min_experience_years"),
            resume_struct.get("total_experience_years"),
        )

        education_score = _score_education(
            jd_struct.get("education_level"),
            resume_struct.get("education_level"),
        )

        cert_score, matched_certs, missing_certs = _score_certifications(
            jd_struct.get("certifications") or [],
            resume_struct.get("certifications") or [],
            embeddings,
        )

        overall_score = round(
            required_score * (WEIGHTS["required_skills"] / 100)
            + preferred_score * (WEIGHTS["preferred_skills"] / 100)
            + projects_score * (WEIGHTS["projects"] / 100)
            + experience_score * (WEIGHTS["experience"] / 100)
            + education_score * (WEIGHTS["education"] / 100)
            + cert_score * (WEIGHTS["certifications"] / 100)
        )

        recommendation = _recommendation(overall_score)

        strengths = (
            list(required_match["matched"])
            + list(preferred_match["matched"])
            + relevant_projects
            + matched_certs
        )
        weaknesses = (
            list(required_match["missing"])
            + missing_certs
        )

        keyword_evidence = _keyword_evidence(
            jd_struct.get("keywords") or [],
            resume_skills_all + [a for a in (resume_struct.get("achievements") or [])],
        )

        summary = _build_summary(
            candidate_name, overall_score, required_match["matched"], required_match["missing"], recommendation
        )

        return {
            "candidate_name": candidate_name,
            "overall_score": overall_score,
            "category_scores": {
                "skills": skills_category_score,
                "projects": round(projects_score),
                "experience": round(experience_score),
                "education": round(education_score),
                "certifications": round(cert_score),
            },
            "score_breakdown": {
                "skills": {
                    "score": skills_category_score,
                    "weight_percent": 55,
                },
                "projects": {
                    "score": round(projects_score),
                    "weight_percent": 15,
                },
                "experience": {
                    "score": round(experience_score),
                    "weight_percent": 15,
                },
                "education": {
                    "score": round(education_score),
                    "weight_percent": 10,
                },
                "certifications": {
                    "score": round(cert_score),
                    "weight_percent": 5,
                },
            },
            "matched_skills": required_match["matched"] + preferred_match["matched"],
            "missing_skills": required_match["missing"] + preferred_match["missing"],
            "matched_certifications": matched_certs,
            "missing_certifications": missing_certs,
            "relevant_projects": relevant_projects,
            "keyword_evidence": keyword_evidence,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendation": recommendation,
            "summary": summary,
            "evidence_chunks": [],
        }

    except Exception as e:
        logger.error(f"Scoring failed for candidate {candidate_name}: {e}")
        return {
            "candidate_name": candidate_name,
            "overall_score": 0,
            "category_scores": {
                "skills": 0, "projects": 0, "experience": 0, "education": 0, "certifications": 0,
            },
            "matched_skills": [],
            "missing_skills": [],
            "matched_certifications": [],
            "missing_certifications": [],
            "relevant_projects": [],
            "keyword_evidence": [],
            "strengths": [],
            "weaknesses": [],
            "recommendation": "Do Not Shortlist",
            "summary": f"⚠️ Scoring failed: {e}",
            "evidence_chunks": [],
        }