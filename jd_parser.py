"""
jd_parser.py
-------------
Turns a raw job-description string into structured facts (required
skills, preferred skills, experience/education requirements, etc.)
using Gemini as a pure extractor. This runs ONCE per screening session
(not once per resume) — the JD doesn't change between candidates, so
re-parsing it for every resume would be a wasted API call.
"""

from prompts import JD_PARSE_PROMPT
from utils import call_gemini_json

JD_FALLBACK = {
    "required_skills": [],
    "preferred_skills": [],
    "responsibilities": [],
    "min_experience_years": None,
    "education_level": None,
    "certifications": [],
    "keywords": [],
}


def parse_job_description(jd_text: str, llm) -> dict:
    """Extract structured requirements from a job description via Gemini."""
    prompt = JD_PARSE_PROMPT.format(jd_text=jd_text)
    result = call_gemini_json(llm, prompt, fallback=JD_FALLBACK)

    # Ensure every expected key exists even if the model omitted one,
    # so downstream scoring code never has to guard against missing keys.
    for key, default in JD_FALLBACK.items():
        result.setdefault(key, default)

    return result
