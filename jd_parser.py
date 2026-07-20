
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
    prompt = JD_PARSE_PROMPT.format(jd_text=jd_text)
    result = call_gemini_json(llm, prompt, fallback=JD_FALLBACK)
    for key, default in JD_FALLBACK.items():
        result.setdefault(key, default)
    return result
