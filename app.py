"""app.py"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import rank_resumes
from resume_parser import extract_resume_text

import cache as cache_module
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

st.title("📄 AI-Powered Resume Screener")
st.caption("ATS-grade RAG pipeline built with LangChain + Gemini + FAISS")

api_key = os.getenv("GOOGLE_API_KEY", "").strip()

with st.sidebar:
    st.markdown(
        "**How it works:**\n"
        "1. Paste a job description\n"
        "2. Upload resumes (PDF/DOCX)\n"
        "3. Click Screen Resumes\n"
        "4. Get ranked, explained ATS reports"
    )
    st.markdown("---")
    if api_key:
        st.success("Gemini API key loaded successfully")
    else:
        st.error(
            f"No Gemini API key found.\n\nAdd GOOGLE_API_KEY to:\n`{ENV_PATH}`\n\n"
            f"then restart the app (env vars only load on startup)."
        )


@st.cache_data(show_spinner=False)
def _extract_cached(file_name: str, file_bytes: bytes) -> str:
    """Cache text extraction per uploaded file's bytes, keyed by name+content."""
    return extract_resume_text(file_name, file_bytes)


col1, col2 = st.columns([1, 1])

with col1:
    job_description = st.text_area(
        "Job Description",
        height=280,
        placeholder="Paste the full job description here...",
    )

with col2:
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} resume(s) uploaded:**")
        for f in uploaded_files:
            st.write(f"- {f.name}")

run_button = st.button("🔍 Screen Resumes", type="primary", use_container_width=True)

if run_button:
    if not api_key:
        st.error("No Gemini API key found. Add GOOGLE_API_KEY to your .env file and restart the app.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        progress = st.progress(0, text="Parsing resumes...")
        parsed_resumes = []
        total_files = len(uploaded_files)

        for i, f in enumerate(uploaded_files):
            try:
                text = _extract_cached(f.name, f.read())
            except Exception as e:
                text = ""
                st.warning(f"Could not parse {f.name}: {e}")
            parsed_resumes.append((f.name, text))
            progress.progress((i + 1) / total_files * 0.3, text=f"Parsing resumes... ({i + 1}/{total_files})")

        progress.progress(0.4, text="Extracting job description requirements...")
        progress.progress(0.55, text="Running ATS scoring pipeline...")

        try:
            pipeline_output = rank_resumes(job_description, parsed_resumes, api_key)
        except Exception as e:
            st.error(f"Something went wrong while calling the Gemini API: {e}")
            pipeline_output = {"jd_struct": {}, "results": [], "jd_error": str(e)}

        progress.progress(0.9, text="Retrieving supporting evidence...")

        if pipeline_output.get("jd_error"):
            st.error(f"Could not parse the job description: {pipeline_output['jd_error']}")

        results = pipeline_output.get("results", [])
        progress.progress(1.0, text="Done.")
        progress.empty()

        if results:
            st.success(f"Screened {len(results)} resume(s).")

            with st.expander("📋 Parsed Job Description Requirements"):
                jd_struct = pipeline_output.get("jd_struct", {})
                st.markdown(f"**Required Skills:** {', '.join(jd_struct.get('required_skills', [])) or '—'}")
                st.markdown(f"**Preferred Skills:** {', '.join(jd_struct.get('preferred_skills', [])) or '—'}")
                st.markdown(f"**Min. Experience:** {jd_struct.get('min_experience_years') or '—'} years")
                st.markdown(f"**Education Level:** {jd_struct.get('education_level') or '—'}")
                st.markdown(f"**Certifications:** {', '.join(jd_struct.get('certifications', [])) or '—'}")
                st.markdown(f"**ATS Keywords:** {', '.join(jd_struct.get('keywords', [])) or '—'}")

            table_data = [
                {
                    "Rank": i + 1,
                    "Candidate": r["candidate_name"],
                    "Overall": r["overall_score"],
                    "Skills": r["category_scores"]["skills"],
                    "Projects": r["category_scores"]["projects"],
                    "Experience": r["category_scores"]["experience"],
                    "Education": r["category_scores"]["education"],
                    "Certs": r["category_scores"]["certifications"],
                    "Recommendation": r.get("recommendation", "—"),
                }
                for i, r in enumerate(results)
            ]
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
            csv = pd.DataFrame(table_data).to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="ats_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("### Detailed Breakdown")
            for i, r in enumerate(results):
                header = f"#{i + 1} — {r['candidate_name']} — {r['overall_score']}/100 — {r.get('recommendation', '')}"
                with st.expander(header):
                    st.markdown(f"**Summary:** {r.get('summary', '')}")

                    breakdown = r.get("score_breakdown", {})
                    cat_cols = st.columns(5)
                    for col, (label, key) in zip(
                        cat_cols,
                        [("Skills", "skills"), ("Projects", "projects"), ("Experience", "experience"),
                         ("Education", "education"), ("Certs", "certifications")],
                    ):
                        entry = breakdown.get(key, {})
                        score = entry.get("score", r.get("category_scores", {}).get(key, 0))
                        weight = entry.get("weight_percent", "—")
                        col.metric(label, f"{score}/100", help=f"Weight: {weight}%")

                    strengths = r.get("strengths", [])
                    if strengths:
                        st.markdown("**Strengths:**")
                        st.write(", ".join(strengths))

                    weaknesses = r.get("weaknesses", [])
                    if weaknesses:
                        st.markdown("**Weaknesses:**")
                        st.write(", ".join(weaknesses))

                    missing = r.get("missing_skills", [])
                    if missing:
                        st.markdown("**Missing Skills:**")
                        st.write(", ".join(missing))

                    matched_certs = r.get("matched_certifications", [])
                    missing_certs = r.get("missing_certifications", [])
                    if matched_certs or missing_certs:
                        st.markdown(
                            f"**Certifications:** matched — {', '.join(matched_certs) or '—'} | "
                            f"missing — {', '.join(missing_certs) or '—'}"
                        )

                    keyword_evidence = r.get("keyword_evidence", [])
                    if keyword_evidence:
                        st.markdown(f"**Additional Keyword Evidence:** {', '.join(keyword_evidence)}")

                    evidence = r.get("evidence_chunks", [])
                    if evidence:
                        with st.expander("Evidence used (retrieved resume chunks)"):
                            for chunk in evidence:
                                st.markdown(f"**[{chunk['section']}]**")
                                st.text(chunk["text"][:500])

# stats = cache_module.cache_stats()
# with st.expander("⚡ Cache Statistics"):
#     st.json(stats)
st.markdown("---")

st.markdown(
    """
<div style="text-align:center; color:gray; font-size:15px; padding:15px;">
<b>Developed by $IDHU</b><br>
B.Tech CSE (AI & ML) | Lovely Professional University<br><br>

<a href="https://github.com/Sidhu-44" target="_blank" style="color: gray; text-decoration: none; font-weight: bold;">GitHub
</a>
&nbsp;|&nbsp;
<a href="https://www.linkedin.com/in/ande-naga-subramanyam/" target="_blank" style="color: gray; text-decoration: none; font-weight: bold;">LinkedIn</a>
</div>
""",
    unsafe_allow_html=True,
)