# AI-Powered Resume Screener — ATS-Grade RAG Pipeline

Ranks and scores resumes against a job description using LangChain,
Gemini, and FAISS — with a deterministic, auditable scoring engine
instead of asking the LLM to invent a score.

## Why this redesign

The original version asked Gemini directly for a 0-100 score, which is
a black box: the same resume could score differently on different runs,
and there was no way to audit *why* a candidate got a given number.

This version splits the job in two:
- **Gemini extracts facts only** — required skills, experience years,
  education level, certifications, etc. — from the JD and each resume,
  as structured JSON. It never rates or scores anything.
- **Python computes the score** — a fixed, documented weighting scheme
  turns those facts into a reproducible 0-100 ATS report.

## Architecture

```
app.py                 Streamlit UI
rag_pipeline.py         Orchestrator: JD parsed once, MMR retrieval per resume, wires everything together
jd_parser.py             Gemini extracts structured JD requirements (once per run)
resume_analyzer.py       Section-aware chunking + Gemini extracts structured resume facts
skill_normalizer.py      Alias table (ML=Machine Learning, etc.) + embedding-based semantic matching
scorer.py                Pure Python — weighted ATS scoring, no LLM calls
prompts.py               All Gemini prompt templates, isolated for easy tuning
utils.py                 Shared helpers: response normalization, safe JSON parsing
resume_parser.py         PDF/DOCX -> plain text extraction
```

## How a screening run works

1. **Parse JD once** — `jd_parser.parse_job_description` extracts required
   skills, preferred skills, min experience, education level, and
   certifications as JSON. This happens once per run, not once per resume.
2. **Embed JD once** — its embedding vector is reused as the retrieval
   query for every candidate, instead of re-embedding the same JD text
   repeatedly.
3. **Per resume:**
   - `resume_analyzer.chunk_resume_by_sections` splits the resume along
     detected headers (Skills, Projects, Experience, Education,
     Certifications...), falling back to fixed-size chunking if no
     clear headers are found.
   - Those chunks are embedded and indexed in FAISS, then retrieved via
     **Max Marginal Relevance (MMR)** against the JD embedding — MMR
     pulls back relevant *and* diverse chunks, so you don't end up with
     six near-duplicate "Skills" chunks and zero "Experience" chunks.
   - `resume_analyzer.parse_resume` extracts structured facts (skills,
     projects, experience, education, certifications) from just those
     retrieved chunks.
   - `scorer.score_resume` computes the ATS report in pure Python:
     - **Skill matching** (`skill_normalizer.match_skills`) first checks
       an extendable alias table (ML ↔ Machine Learning, JS ↔ JavaScript,
       GitHub ↔ Git, etc.), then falls back to embedding-based semantic
       similarity for anything left unmatched — batched into one
       embedding call per resume, not one call per skill pair.
     - **Weights:** Required Skills 50%, Projects 15%, Experience 15%,
       Education 10%, Certifications 5%, Preferred Skills 5%.
     - Every score component is explainable: matched/missing skills,
       matched/missing certifications, and the retrieved evidence chunks
       are all attached to the result.
4. Candidates are ranked by `overall_score`, descending.

## Setup

```bash
pip install -r requirements.txt
```

Get a free Gemini API key at https://aistudio.google.com/apikey, then:

- Copy `.env.example` to `.env` and paste your key into `GOOGLE_API_KEY`.

The app only reads the key from `.env` — there's no manual entry field in
the UI, so restart the app after changing `.env`.

If a model name ever gets deprecated, you can override it without touching
code by adding to `.env`:
```
GEMINI_CHAT_MODEL=gemini-flash-latest
GEMINI_EMBED_MODEL=gemini-embedding-001
```

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Extending

- **Add a skill synonym:** add a line to `CANONICAL_SKILL_MAP` in
  `skill_normalizer.py`.
- **Change scoring weights:** edit `WEIGHTS` at the top of `scorer.py`.
  They're expressed as points out of 100, so they should sum to 100.
- **Tune retrieval:** `MMR_K`, `MMR_FETCH_K`, and `MMR_LAMBDA` in
  `rag_pipeline.py` control how many chunks are retrieved and the
  relevance/diversity tradeoff.
- **Adjust prompts:** everything Gemini is asked lives in `prompts.py`,
  separate from the code that calls it.

## Notes on API usage

Per screening run: 1 chat call + 1 embedding call for the JD (once,
not per resume). Per resume: 1 chat call (fact extraction) + 1-2
embedding calls (resume chunk embeddings, plus a batched semantic
skill-matching call only when the alias table doesn't already resolve
a skill). No LLM call is ever used to produce a score.

