"""skill_normalizer.py"""

import re
from typing import Dict, List, Optional

import numpy as np

from cache import get_skill_embedding, save_skill_embedding
from utils import get_logger, hash_text

logger = get_logger()

SEMANTIC_MATCH_THRESHOLD = 0.86

CANONICAL_SKILL_MAP: Dict[str, str] = {

    # =========================
    # AI / ML
    # =========================
    "ml": "machine learning",
    "machine learning": "machine learning",

    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",

    "genai": "generative ai",
    "gen ai": "generative ai",
    "generative ai": "generative ai",

    "llm": "large language model",
    "llms": "large language model",
    "large language model": "large language model",
    "large language models": "large language model",

    "nlp": "natural language processing",
    "natural language processing": "natural language processing",

    "cv": "computer vision",
    "computer vision": "computer vision",

    "rag": "retrieval augmented generation",
    "retrieval augmented generation": "retrieval augmented generation",

    "langchain": "langchain",
    "llamaindex": "llamaindex",
    "faiss": "faiss",
    "chromadb": "chromadb",
    "pinecone": "pinecone",

    "tensorflow": "tensorflow",
    "tf": "tensorflow",

    "pytorch": "pytorch",
    "torch": "pytorch",

    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "scikit-learn": "scikit-learn",

    "numpy": "numpy",
    "pandas": "pandas",

    # =========================
    # Frontend
    # =========================
    "html": "html5",
    "html5": "html5",

    "css": "css3",
    "css3": "css3",

    "sass": "sass",
    "scss": "sass",

    "javascript": "javascript",
    "js": "javascript",

    "typescript": "typescript",
    "ts": "typescript",

    "react": "react.js",
    "reactjs": "react.js",
    "react.js": "react.js",

    "next": "next.js",
    "nextjs": "next.js",
    "next.js": "next.js",

    "redux": "redux",
    "redux toolkit": "redux",

    "tailwind": "tailwind css",
    "tailwindcss": "tailwind css",
    "tailwind css": "tailwind css",

    "bootstrap": "bootstrap",

    "material ui": "material ui",
    "mui": "material ui",

    "chakra ui": "chakra ui",

    "responsive ui": "responsive design",
    "responsive web design": "responsive design",
    "responsive design": "responsive design",

    "vite": "vite",

    # =========================
    # Backend
    # =========================
    "node": "node.js",
    "nodejs": "node.js",
    "node.js": "node.js",

    "express": "express.js",
    "expressjs": "express.js",
    "express.js": "express.js",

    "django": "django",
    "flask": "flask",
    "fastapi": "fastapi",

    "spring": "spring boot",
    "spring boot": "spring boot",

    "rest api": "rest apis",
    "rest apis": "rest apis",
    "restful api": "rest apis",
    "restful apis": "rest apis",

    "graphql": "graphql",

    "jwt": "jwt",

    "oauth": "oauth",
    "oauth2": "oauth",

    # =========================
    # Database
    # =========================
    "db": "database",
    "database": "database",

    "sql": "sql",

    "mysql": "mysql",

    "postgres": "postgresql",
    "postgresql": "postgresql",

    "mongo": "mongodb",
    "mongodb": "mongodb",

    "sqlite": "sqlite",

    "redis": "redis",

    "oracle": "oracle",

    "sql server": "sql server",
    "mssql": "sql server",

    # =========================
    # Cloud
    # =========================
    "aws": "amazon web services",
    "amazon web services": "amazon web services",

    "azure": "microsoft azure",
    "microsoft azure": "microsoft azure",

    "azure ad": "azure entra id",
    "azure entra id": "azure entra id",

    "gcp": "google cloud platform",
    "google cloud platform": "google cloud platform",

    # =========================
    # DevOps
    # =========================
    "docker": "docker",

    "k8s": "kubernetes",
    "kubernetes": "kubernetes",

    "jenkins": "jenkins",

    "github": "git",
    "git": "git",

    "github actions": "github actions",

    "gitlab ci": "gitlab ci",

    "ci/cd": "ci cd",
    "ci cd": "ci cd",
    "continuous integration": "ci cd",
    "continuous deployment": "ci cd",

    "terraform": "terraform",

    "ansible": "ansible",

    # =========================
    # Testing
    # =========================
    "pytest": "pytest",

    "jest": "jest",

    "react testing library": "react testing library",
    "rtl": "react testing library",

    "selenium": "selenium",

    "cypress": "cypress",

    "playwright": "playwright",

    "junit": "junit",

    # =========================
    # Programming Languages
    # =========================
    "python": "python",

    "java": "java",

    "c": "c",

    "c++": "cpp",
    "cpp": "cpp",

    "c#": "csharp",
    "csharp": "csharp",

    "go": "golang",
    "golang": "golang",

    "rust": "rust",

    "kotlin": "kotlin",

    "swift": "swift",

    "php": "php",

    # =========================
    # Tools
    # =========================
    "streamlit": "streamlit",

    "postman": "postman",

    "swagger": "swagger",

    "gitlab": "gitlab",

    "jira": "jira",

    "figma": "figma",

    # =========================
    # Soft Skills
    # =========================
    "communication skills": "communication",
    "communication": "communication",

    "problem solving": "problem solving",
    "problem-solving": "problem solving",

    "team work": "teamwork",
    "teamwork": "teamwork",

    "leadership": "leadership",

    "critical thinking": "critical thinking",

    "adaptability": "adaptability",

    "time management": "time management",

    "agile": "agile",
    "agile methodology": "agile",

    "scrum": "scrum",

    "kanban": "kanban",
}

EDUCATION_LEVELS: Dict[str, int] = {
    "high_school": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "phd": 5,
}

EDUCATION_ALIASES: Dict[str, str] = {
    "high school": "high_school",
    "highschool": "high_school",
    "hs": "high_school",
    "secondary school": "high_school",
    "diploma": "associate",
    "associate": "associate",
    "associate degree": "associate",
    "aa": "associate",
    "as": "associate",
    "bachelor": "bachelor",
    "bachelors": "bachelor",
    "bachelor's": "bachelor",
    "btech": "bachelor",
    "b tech": "bachelor",
    "be": "bachelor",
    "bachelor of technology": "bachelor",
    "bachelor of engineering": "bachelor",
    "bachelor of science": "bachelor",
    "bachelor of arts": "bachelor",
    "bsc": "bachelor",
    "bs": "bachelor",
    "ba": "bachelor",
    "undergraduate": "bachelor",
    "master": "master",
    "masters": "master",
    "master's": "master",
    "mtech": "master",
    "m tech": "master",
    "me": "master",
    "ms": "master",
    "msc": "master",
    "ma": "master",
    "master of technology": "master",
    "master of engineering": "master",
    "master of science": "master",
    "master of arts": "master",
    "mba": "master",
    "postgraduate": "master",
    "phd": "phd",
    "ph d": "phd",
    "doctorate": "phd",
    "doctoral": "phd",
}


def _clean_key(text: str) -> str:
    """Lowercase and strip punctuation/extra whitespace for dictionary lookups."""
    cleaned = re.sub(r"[.\-']", "", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_skill(skill: str) -> str:
    """Normalize a skill by cleaning punctuation and mapping aliases."""
    key = _clean_key(skill or "")
    return CANONICAL_SKILL_MAP.get(key, key)


def normalize_skill_list(skills: List[str]) -> List[str]:
    """Normalize a list of skills, de-duplicating while preserving order."""
    seen = set()
    normalized = []
    for s in skills or []:
        norm = normalize_skill(s)
        if norm and norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


def normalize_education(level: Optional[str]) -> Optional[str]:
    """
    Map a free-form education string to one of: high_school, associate,
    bachelor, master, phd. Returns None if unrecognized rather than guessing.
    """
    if not level:
        return None
    key = _clean_key(level)
    if key in EDUCATION_LEVELS:
        return key
    return EDUCATION_ALIASES.get(key)


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def embed_with_cache(texts: List[str], embeddings) -> np.ndarray:
    """
    Embed a list of texts, reusing cached vectors keyed by content hash.
    Only texts not already cached are sent to the API, in a single
    batched call. Raises on API failure so callers can degrade gracefully.
    """
    if not texts:
        return np.zeros((0, 0))

    hashes = [hash_text(t) for t in texts]
    vectors: List[Optional[List[float]]] = [None] * len(texts)
    uncached_indices = []

    for i, h in enumerate(hashes):
        cached = get_skill_embedding(h)
        if cached is not None:
            vectors[i] = cached
        else:
            uncached_indices.append(i)

    if uncached_indices:
        to_embed = [texts[i] for i in uncached_indices]
        try:
            new_vectors = embeddings.embed_documents(to_embed)
        except Exception as e:
            logger.warning(f"Embedding call failed for {len(to_embed)} text(s): {e}")
            raise
        for idx, vec in zip(uncached_indices, new_vectors):
            vectors[idx] = vec
            save_skill_embedding(hashes[idx], vec)

    logger.info(
        f"embed_with_cache: {len(texts) - len(uncached_indices)} cache hit(s), "
        f"{len(uncached_indices)} embedded fresh"
    )
    return np.array(vectors)


def match_skills(jd_skills: List[str], resume_skills: List[str], embeddings) -> Dict[str, List[str]]:
    """
    Match JD skills against resume skills: exact/alias match first, then
    cached semantic similarity for anything left unmatched.
    Returns {"matched": [...], "missing": [...]} using original JD skill strings.
    """
    if not jd_skills:
        return {"matched": [], "missing": []}

    jd_norm = [normalize_skill(s) for s in jd_skills]
    resume_norm_set = set(normalize_skill(s) for s in resume_skills)

    matched: List[str] = []
    unmatched_indices: List[int] = []

    for i, (orig, norm) in enumerate(zip(jd_skills, jd_norm)):
        if norm in resume_norm_set:
            matched.append(orig)
        else:
            unmatched_indices.append(i)

    missing: List[str] = []

    if unmatched_indices and resume_skills:
        remaining_jd = [jd_skills[i] for i in unmatched_indices]
        try:
            combined = remaining_jd + resume_skills
            vectors = embed_with_cache(combined, embeddings)
            jd_vecs = vectors[: len(remaining_jd)]
            resume_vecs = vectors[len(remaining_jd):]
            sim_matrix = _cosine_sim_matrix(jd_vecs, resume_vecs)
            best_sim_per_jd = sim_matrix.max(axis=1)

            for local_i, orig in enumerate(remaining_jd):
                if best_sim_per_jd[local_i] >= SEMANTIC_MATCH_THRESHOLD:
                    matched.append(orig)
                else:
                    missing.append(orig)
        except Exception as e:
            logger.warning(f"Semantic skill matching failed, treating remainder as missing: {e}")
            missing.extend(remaining_jd)
    else:
        missing.extend(jd_skills[i] for i in unmatched_indices)

    return {"matched": matched, "missing": missing}