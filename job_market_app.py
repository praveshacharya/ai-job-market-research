"""
job_market_app.py

Streamlit prototype: AI Tool for Market Research in the Job Industry

How to run:
1. Save this file as job_market_app.py
2. Install dependencies:
   pip install streamlit pandas numpy matplotlib spacy
   python -m spacy download en_core_web_sm
3. Run:
   streamlit run job_market_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import json
import textwrap

# Optional: if you plan to call an LLM for summarization, uncomment related imports and implement call
# import requests

# ---------- Configuration ----------
st.set_page_config(page_title="AI Job Market Research Tool", layout="wide")
TITLE = "AI Tool for Market Research — Job Industry (Prototype)"

# ---------- Helpers & Sample Data ----------

SAMPLE_POSTINGS = {
    "Data Scientist": [
        ("Entry",
         "Looking for an Entry-level Data Scientist. Must know Python, SQL, basic statistics. Experience with pandas and exploratory data analysis. Good communication and teamwork skills required."),
        ("Mid",
         "Mid-level Data Scientist required. Proficient in Python, machine learning (sklearn), SQL, and model evaluation. Experience with cloud platforms (AWS/GCP). Strong problem solving and stakeholder communication."),
        ("Senior",
         "Senior Data Scientist needed. Deep experience with Python, ML, deep learning, model deployment, SQL, Spark. Leadership experience, mentoring, and cross-functional collaboration. Familiar with MLOps."),
    ],
    "Data Analyst": [
        ("Entry",
         "Entry Data Analyst. Must be strong in Excel, SQL, basic Python (pandas), data visualization (Tableau/Power BI). Detail-oriented, good communication."),
        ("Mid",
         "Data Analyst with SQL, Python, dashboarding experience. Able to translate business questions into analyses. Familiar with A/B testing."),
    ],
    "Machine Learning Engineer": [
        ("Mid",
         "ML Engineer with Python, TensorFlow/PyTorch, model deployment (Docker, Kubernetes), CI/CD pipelines, cloud (AWS/GCP). Requires collaboration with data scientists and software engineers."),
        ("Senior",
         "Senior ML Engineer: design and productionize ML systems, strong software engineering, mentoring, architecture decisions. Experience with distributed training, model monitoring."),
    ],
    "AI Engineer": [
        ("Mid",
         "AI Engineer with experience in LLMs, prompt engineering, Python, APIs, and cloud deployment. Must have collaborative and product-minded skills."),
    ],
    "Business Analyst": [
        ("Entry",
         "Business Analyst - entry. Strong Excel, stakeholder communication, process mapping. Basic SQL knowledge helpful."),
    ],
    "Data Engineer": [
        ("Mid",
         "Data Engineer: ETL design, SQL, Python/Scala, Spark, data warehousing, cloud (AWS/GCP). Works closely with analytics teams."),
    ],
}

# Predefined skill lists (expand as needed)
TECH_SKILLS = [
    "python", "sql", "excel", "tableau", "power bi", "pandas", "spark",
    "aws", "gcp", "azure", "docker", "kubernetes", "tensorflow", "pytorch",
    "machine learning", "deep learning", "etl", "hadoop", "spark", "scala",
    "github", "ci/cd", "bigquery", "sql server", "redshift"
]
SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "problem solving", "stakeholder",
    "collaboration", "mentoring", "presentation", "critical thinking", "adaptability",
    "time management"
]

# simple normalization
def norm(text):
    return re.sub(r'[^a-z0-9\s\+\-/\.]', ' ', text.lower())

# Extract skills using keyword matching + fuzzy-ish matching for multi-word skills
def extract_skills_from_text(text):
    text_l = norm(text)
    found_tech = set()
    found_soft = set()
    for s in TECH_SKILLS:
        if re.search(r'\b' + re.escape(s) + r'\b', text_l):
            found_tech.add(s)
    for s in SOFT_SKILLS:
        if re.search(r'\b' + re.escape(s) + r'\b', text_l):
            found_soft.add(s)
    # Heuristics: look for patterns like "experience with X" or "familiar with X"
    extra = re.findall(r'(experience with|familiar with|knowledge of|proficient in|must know|must be strong in)\s+([a-z0-9\-\s/\.,&]+)', text_l)
    for _, tail in extra:
        # split tail by commas and 'and'
        parts = re.split(r',| and | & |/|;', tail)
        for p in parts:
            p = p.strip()
            if len(p) < 2: 
                continue
            for s in TECH_SKILLS:
                if s in p and s not in found_tech:
                    found_tech.add(s)
            for s in SOFT_SKILLS:
                if s in p and s not in found_soft:
                    found_soft.add(s)
    return sorted(found_tech), sorted(found_soft)

# ---------- Data fetching (prototype) ----------
@st.cache_data(ttl=60*60)
def fetch_job_postings(job_title: str, level: str, n: int = 10):
    """
    Prototype fetcher:
    - If you connect a real job API, replace this function's body.
    - For now, it uses SAMPLE_POSTINGS as fallback.
    Returns a list of job descriptions (strings).
    """
    postings = []
    sample_list = SAMPLE_POSTINGS.get(job_title, [])
    # Prefer exact level matches, otherwise mix
    exact = [desc for lvl, desc in sample_list if lvl.lower() == level.lower()]
    if exact:
        # replicate variations to reach count n
        for i in range(n):
            postings.append(exact[i % len(exact)])
    else:
        # collect any postings for the job and/or similar roles
        if sample_list:
            for i in range(n):
                postings.append(sample_list[i % len(sample_list)][1])
        else:
            # fallback: generic message
            for i in range(n):
                postings.append(f"{job_title} role description example. Required skills: Python, SQL. Soft skills: communication, teamwork.")
    return postings

# ---------- Analysis ----------
@st.cache_data(ttl=60*30)
def analyze_postings(postings):
    skill_counts = Counter()
    tech_counts = Counter()
    soft_counts = Counter()
    all_text = " ".join(postings)

    for p in postings:
        techs, softs = extract_skills_from_text(p)
        for t in techs:
            tech_counts[t] += 1
            skill_counts[t] += 1
        for s in softs:
            soft_counts[s] += 1
            skill_counts[s] += 1

    total_skills_found = sum(skill_counts.values()) or 1
    tech_total = sum(tech_counts.values())
    soft_total = sum(soft_counts.values())

    # Compose a DataFrame of skills
    rows = []
    for k, v in skill_counts.most_common():
        kind = "tech" if k in TECH_SKILLS else ("soft" if k in SOFT_SKILLS else "other")
        rows.append({"skill": k, "count": v, "kind": kind, "pct_of_skills": v / total_skills_found * 100})
    df_skills = pd.DataFrame(rows)
    summary = {
        "total_postings": len(postings),
        "unique_skills_found": len(df_skills),
        "tech_count": tech_total,
        "soft_count": soft_total,
        "tech_pct_of_skills": 100 * tech_total / total_skills_found,
        "soft_pct_of_skills": 100 * soft_total / total_skills_found,
    }
    return df_skills, summary, all_text

# ---------- LLM Summarization placeholder ----------
def summarize_with_llm(text, job_title, level):
    """
    Placeholder wrapper for an LLM summarization call.
    Replace with actual API call (Google AI Studio / OpenAI) if you have keys.
    For this prototype we return a deterministic summary.
    """
    # Example of what you might send to the LLM:
    # prompt = f"Given these job descriptions for {job_title} ({level}), summarize top skills and recommended actions for a candidate."
    # call the API -> return response
    # For now, return a short auto-generated summary:
    token_preview = textwrap.shorten(text.replace("\n"," "), width=800, placeholder=" ...")
    summary = f"Summary (simulated): For {job_title} ({level}), commonly requested technical skills include Python and SQL; employers often ask for data visualization and cloud familiarity. Soft skills commonly requested include communication and teamwork. Example excerpt: {token_preview[:350]}"
    return summary

# ---------- Streamlit UI ----------
st.title(TITLE)
st.markdown(
    """
    Prototype to analyze job postings and extract top technical and soft skills.
    - Uses sample data by default so the app runs offline.
    - To integrate real job APIs or LLM summarizers, replace the fetch_job_postings and summarize_with_llm functions.
    """
)

# Sidebar controls
with st.sidebar:
    st.header("Analyze Job Market")
    job_titles = sorted(list(SAMPLE_POSTINGS.keys()))
    job_titles = job_titles + ["Data Scientist", "Data Analyst", "Business Analyst", "Machine Learning Engineer", "AI Engineer", "Data Engineer", "Business Intelligence Specialist"]
    selected_job = st.selectbox("Job Title", options=sorted(set(job_titles)))
    selected_level = st.selectbox("Career Level", options=["Entry", "Mid", "Senior", "Lead/Manager"])
    num_postings = st.slider("Number of postings to analyze (prototype)", min_value=1, max_value=50, value=6)
    run_analysis = st.button("Run Analysis")

# Main area: show sample postings and results
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Selected Inputs")
    st.write("**Job Title:**", selected_job)
    st.write("**Career Level:**", selected_level)
    st.write("**Postings to analyze:**", num_postings)

    st.subheader("Sample / Raw Postings")
    postings = fetch_job_postings(selected_job, selected_level, n=num_postings)
    # show first few postings
    for i, p in enumerate(postings[:6], start=1):
        st.markdown(f"**Posting {i}:** {p}")

with col2:
    if run_analysis:
        with st.spinner("Analyzing postings..."):
            df_skills, summary, all_text = analyze_postings(postings)

        st.subheader("Summary Metrics")
        st.metric("Total Postings", summary["total_postings"])
        st.metric("Unique skills found", summary["unique_skills_found"])
        st.metric("Tech % of skills", f"{summary['tech_pct_of_skills']:.1f}%")
        st.metric("Soft % of skills", f"{summary['soft_pct_of_skills']:.1f}%")

        st.subheader("Top Skills (by frequency)")
        if df_skills.empty:
            st.info("No skills found in sample postings. Try a different job title or add real postings.")
        else:
            # Show table
            st.dataframe(df_skills.head(20).sort_values("count", ascending=False).reset_index(drop=True))

            # Chart: top tech and soft
            top_k = df_skills.head(10).sort_values("count", ascending=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(top_k["skill"], top_k["count"])
            ax.set_xlabel("Count")
            ax.set_title("Top skills (sample)")
            st.pyplot(fig)

        st.subheader("AI-generated Summary (simulated)")
        summary_text = summarize_with_llm(all_text, selected_job, selected_level)
        st.write(summary_text)

        # Downloads: CSV & JSON
        st.subheader("Download Results")
        csv = df_skills.to_csv(index=False)
        st.download_button("Download skills CSV", csv, file_name=f"{selected_job}_{selected_level}_skills.csv")
        st.download_button("Download raw postings (JSON)", json.dumps(postings, indent=2), file_name=f"{selected_job}_{selected_level}_postings.json", mime="application/json")

        # Small recommendations based on counts
        st.subheader("Actionable Recommendations (automated heuristics)")
        recs = []
        # if python is common
        if any(df_skills['skill'].str.contains('python', case=False, na=False)):
            recs.append("Strengthen Python (pandas, numpy) and demonstrate projects in a portfolio.")
        if any(df_skills['skill'].str.contains('sql', case=False, na=False)):
            recs.append("Practice SQL and be ready to solve real SQL queries; include a SQL-based sample in portfolio.")
        if summary["tech_pct_of_skills"] > 60:
            recs.append("Focus on technical skill depth (coding, cloud) — employers emphasize hard skills for this role.")
        if summary["soft_pct_of_skills"] > 30:
            recs.append("Highlight soft skills (communication, teamwork) on your resume and interviews.")
        if not recs:
            recs = ["Consider adding relevant technical examples to your resume and practice explaining your impact."]

        for r in recs:
            st.write("- " + r)

    else:
        st.info("Configure inputs on the left and click **Run Analysis** to analyze sample job postings.")

# ---------- Optional advanced features guide ----------
st.markdown("---")
st.subheader("How to replace sample data with real job postings / LLM summarization")
st.markdown(
    """
    1. **Connect a job-postings API** (e.g., Adzuna, Indeed, LinkedIn if you have access):
       - Replace `fetch_job_postings()` with a function that calls the API, paginates results, and returns job descriptions.
       - Store API keys in Streamlit secrets or environment variables.
    2. **Use an LLM for better summaries & skill extraction**:
       - Implement `summarize_with_llm()` to call Google AI Studio or OpenAI.
       - Be sure to handle rate limits and chunk long text.
    3. **Improve skill extraction**:
       - Replace keyword matching with spaCy named-entity recognition or a fine-tuned classifier.
    4. **Add trend analysis**:
       - Save analyzed results in a DB and allow time-series queries to show how demand changes over months.
    """
)

st.markdown("### Notes")
st.markdown(
    """
    - This is a **prototype** for demonstration and class project use.
    - If you add real API calls, store keys outside the code and **never** commit secrets to source control.
    - To use the app locally, run `streamlit run job_market_app.py` and open the URL Streamlit prints (usually http://localhost:8501).
    """
)
