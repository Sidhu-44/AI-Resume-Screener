"""resume_analyzer.py"""

import re
from datetime import date
from typing import Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from cache import get_resume, save_resume
from prompts import RESUME_PARSE_PROMPT
from utils import Timer, call_gemini_json, get_logger, hash_text

logger = get_logger()

RESUME_FALLBACK: Dict = {
    "skills": [],
    "tools": [],
    "programming_languages": [],
    "frameworks": [],
    "databases": [],
    "cloud_platforms": [],
    "projects": [],
    "experience_entries": [],
    "education": [],
    "education_level": None,
    "education_field": None,
    "certifications": [],
    "achievements": [],
    "ats_keywords": []
}


SECTION_HEADERS = [
    "summary", "objective", "profile",
    "skills", "technical skills", "core competencies",
    "projects", "personal projects", "academic projects",
    "experience", "work experience", "professional experience", "employment history",
    "education", "academic background",
    "certifications", "certificates", "licenses",
    "tools", "technologies",
]

_HEADER_PATTERN = re.compile(
    r"^\s*(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_FALLBACK_CHUNK_SIZE = 800
_FALLBACK_CHUNK_OVERLAP = 100

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_CURRENT_MARKERS = {"present", "current", "currently", "now", "ongoing", "till date", "to date"}

_MONTH_YEAR_RE = re.compile(
    r"^\s*([A-Za-z]{3,9})\.?\s+(\d{4})\s*$"
)
_NUMERIC_DATE_RE = re.compile(
    r"^\s*(\d{1,2})[/\-](\d{4})\s*$"
)
_ISO_DATE_RE = re.compile(
    r"^\s*(\d{4})-(\d{1,2})(?:-(\d{1,2}))?\s*$"
)
_YEAR_ONLY_RE = re.compile(r"^\s*(\d{4})\s*$")

_DURATION_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*year", re.IGNORECASE)
_DURATION_MONTHS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*month", re.IGNORECASE)


def chunk_resume_by_sections(resume_text: str) -> List[Tuple[str, str]]:
    if not resume_text:
        return [("general", "")]

    matches = list(_HEADER_PATTERN.finditer(resume_text))

    if len(matches) < 2:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_FALLBACK_CHUNK_SIZE,
            chunk_overlap=_FALLBACK_CHUNK_OVERLAP,
        )
        chunks = splitter.split_text(resume_text) or [resume_text]
        return [("general", c) for c in chunks]

    sections: List[Tuple[str, str]] = []
    for i, match in enumerate(matches):
        header = match.group(1).strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(resume_text)
        body = resume_text[start:end].strip()
        if body:
            sections.append((header, body))

    preamble = resume_text[: matches[0].start()].strip()
    if preamble:
        sections.insert(0, ("header", preamble))

    return sections or [("general", resume_text)]


def _parse_date_token(token: Optional[str], is_end: bool) -> Optional[date]:
    """Parse a single date token in various formats; None if unparseable."""
    if not token:
        return None
    text = token.strip()
    if not text:
        return None
    if text.lower() in _CURRENT_MARKERS:
        return date.today() if is_end else None

    m = _MONTH_YEAR_RE.match(text)
    if m:
        month_name, year = m.group(1).lower(), int(m.group(2))
        month = _MONTHS.get(month_name)
        if month:
            return date(year, month, 1)

    m = _NUMERIC_DATE_RE.match(text)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return date(year, month, 1)

    m = _ISO_DATE_RE.match(text)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        day = int(m.group(3)) if m.group(3) else 1
        try:
            return date(year, month, day)
        except ValueError:
            return date(year, month, 1)

    m = _YEAR_ONLY_RE.match(text)
    if m:
        year = int(m.group(1))
        return date(year, 12, 31) if is_end else date(year, 1, 1)

    return None


def _duration_text_to_years(duration_text: Optional[str]) -> Optional[float]:
    """Fallback: parse an approximate duration from free text like '1.5 years', '6 months'."""
    if not duration_text:
        return None
    years = 0.0
    found = False
    y_match = _DURATION_YEARS_RE.search(duration_text)
    if y_match:
        years += float(y_match.group(1))
        found = True
    m_match = _DURATION_MONTHS_RE.search(duration_text)
    if m_match:
        years += float(m_match.group(1)) / 12.0
        found = True
    return years if found else None


def _entry_date_range(entry: Dict) -> Optional[Tuple[date, date]]:
    """Resolve one experience entry to a (start, end) date range, or None."""
    start = _parse_date_token(entry.get("start_date"), is_end=False)
    end = _parse_date_token(entry.get("end_date"), is_end=True)

    if start and end and end >= start:
        return start, end
    return None


def compute_experience_years(experience_entries: List[Dict]) -> Optional[float]:
    """
    Compute total professional experience in years from discrete entries,
    merging overlapping/concurrent roles so they aren't double-counted.
    Falls back to summing parsed duration_text for entries without usable
    dates. Returns None if no entry yields any usable information.
    """
    if not experience_entries:
        return None

    ranges: List[Tuple[date, date]] = []
    fallback_years = 0.0
    any_signal = False

    for entry in experience_entries:
        rng = _entry_date_range(entry)
        if rng:
            ranges.append(rng)
            any_signal = True
        else:
            approx = _duration_text_to_years(entry.get("duration_text"))
            if approx is not None:
                fallback_years += approx
                any_signal = True

    if not any_signal:
        return None

    merged_years = 0.0
    if ranges:
        ranges.sort(key=lambda r: r[0])
        merged: List[Tuple[date, date]] = [ranges[0]]
        for start, end in ranges[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        for start, end in merged:
            merged_years += (end - start).days / 365.25

    total = round(merged_years + fallback_years, 2)
    return max(0.0, total)


def parse_resume(candidate_name: str, resume_text: str, llm) -> Dict:
    if not resume_text or not resume_text.strip():
        logger.warning(f"Empty resume text for candidate: {candidate_name}")
        result = dict(RESUME_FALLBACK)
        result["total_experience_years"] = None
        return result

    resume_hash = hash_text(resume_text)
    cached = get_resume(resume_hash)
    if cached is not None:
        logger.info(f"Resume cache hit for {candidate_name} ({resume_hash[:12]}...)")
        result = dict(cached)
    else:
        with Timer(logger, f"Gemini resume extraction: {candidate_name}"):
            prompt = RESUME_PARSE_PROMPT.format(candidate_name=candidate_name, resume_text=resume_text)
            result = call_gemini_json(llm, prompt, fallback=RESUME_FALLBACK)

        if "_error" in result:
            logger.warning(f"Resume extraction failed for {candidate_name}: {result['_error']}")
            return result

        for key, default in RESUME_FALLBACK.items():
            result.setdefault(key, default)

        save_resume(resume_hash, result)

    result["total_experience_years"] = compute_experience_years(result.get("experience_entries") or [])
    return result