"""
AI Resume Screening Agent - Main Streamlit Application
A simple, fast, production-grade web application for candidate resume screening, NLP matching, ranking, and analytics.
100% self-contained & offline-capable (No API keys required).
"""

import os
import io
import json
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    RECOMMENDATION_COLORS, JOB_DESC_DIR, RESUMES_DIR, OUTPUT_DIR
)
from parser import DocumentParser
from extractor import Extractor
from similarity import SimilarityEngine
from scorer import CandidateScorer
from recommender import CandidateRecommender
from agent import RecruiterAIAgent
from utils import (
    export_to_csv, export_to_json, generate_candidate_pdf_report,
    highlight_keywords_in_text, create_sample_dataset
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure sample dataset exists
create_sample_dataset()

# Page Setup
st.set_page_config(
    page_title="AI Resume Screening Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Aesthetic
CUSTOM_CSS = """
<style>
    .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 50%, #3B82F6 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(to right, #FFFFFF, #93C5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(226, 232, 240, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 800;
        color: #3B82F6;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .skill-chip-match {
        display: inline-block;
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .skill-chip-missing {
        display: inline-block;
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    .badge-highly-recommended {
        background-color: #10B981;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
    }
    .badge-recommended {
        background-color: #3B82F6;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
    }
    .badge-consider {
        background-color: #F59E0B;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
    }
    .badge-not-recommended {
        background-color: #EF4444;
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Session State Initialization
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "jd_info" not in st.session_state:
    st.session_state.jd_info = {}
if "screening_results" not in st.session_state:
    st.session_state.screening_results = []
if "parsed_resumes" not in st.session_state:
    st.session_state.parsed_resumes = []
if "is_screened" not in st.session_state:
    st.session_state.is_screened = False
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I am your **AI Recruiter Assistant**. Ask me to evaluate candidates, shortlist applicants, look up resume details, or schedule interviews!"}
    ]


# Sidebar Configuration
with st.sidebar:
    st.title("🎯 Control Panel")
    st.success("⚡ 100% Self-Contained Engine (No API Key Required)")

    st.markdown("---")
    st.subheader("Demo Mode")
    if st.button("⚡ Load Demo Dataset", type="primary", use_container_width=True):
        sample_jd_file = os.path.join(JOB_DESC_DIR, "Senior_AI_Engineer_JD.txt")
        if os.path.exists(sample_jd_file):
            st.session_state.jd_text = DocumentParser.parse_document(sample_jd_file, "Senior_AI_Engineer_JD.txt")
            
        resumes_files = [os.path.join(RESUMES_DIR, f) for f in os.listdir(RESUMES_DIR)]
        
        extractor = Extractor()
        similarity_engine = SimilarityEngine()
        scorer = CandidateScorer()
        recommender = CandidateRecommender()

        st.session_state.jd_info = extractor.extract_job_description_info(st.session_state.jd_text)

        results = []
        resume_texts = []
        parsed_resumes = []

        for r_path in resumes_files:
            r_name = os.path.basename(r_path)
            r_text = DocumentParser.parse_document(r_path, r_name)
            if r_text:
                r_info = extractor.extract_resume_info(r_text, r_name)
                parsed_resumes.append(r_info)
                resume_texts.append(r_text)

        if resume_texts and st.session_state.jd_text:
            sem_scores = similarity_engine.compute_semantic_similarity(st.session_state.jd_text, resume_texts)
            for idx, r_info in enumerate(parsed_resumes):
                eval_res = scorer.evaluate_candidate(r_info, st.session_state.jd_info, sem_scores[idx])
                report = recommender.generate_report(r_info, st.session_state.jd_info, eval_res)
                results.append(report)

            results.sort(key=lambda x: x["overall_score"], reverse=True)
            st.session_state.screening_results = results
            st.session_state.parsed_resumes = parsed_resumes
            st.session_state.is_screened = True

            export_to_csv(results)
            export_to_json(results)
            st.toast("✅ Demo Dataset Loaded & Screened Successfully!", icon="🚀")

    st.markdown("---")
    st.caption("AI Resume Screening Agent v1.0.0")


# App Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🎯 AI Resume Screening Agent</h1>
    <p class="header-subtitle">Automated Multi-Format Screening, NLP Semantic Matching & Recruiter Assistant</p>
</div>
""", unsafe_allow_html=True)


# Main Tabs Navigation
tab_upload, tab_rankings, tab_details, tab_analytics, tab_agent = st.tabs([
    "📥 1. Upload & Setup",
    "🏆 2. Candidate Rankings",
    "📄 3. Candidate Details",
    "📊 4. Analytics & Insights",
    "💬 5. AI Recruiter Assistant"
])


# ==========================================
# TAB 1: UPLOAD & SETUP
# ==========================================
with tab_upload:
    st.subheader("Job Description & Resumes Ingestion")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📋 1. Job Description (JD)")
        jd_file = st.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="jd_uploader")
        jd_text_input = st.text_area("Or Paste Job Description Text Here", value=st.session_state.jd_text, height=220, placeholder="Paste JD requirements, required skills, experience level...")

        if jd_file:
            st.session_state.jd_text = DocumentParser.parse_document(jd_file, jd_file.name)
        elif jd_text_input:
            st.session_state.jd_text = jd_text_input

        if st.session_state.jd_text:
            st.success(f"✅ Job Description Loaded ({len(st.session_state.jd_text)} characters)")

    with col2:
        st.markdown("### 📄 2. Candidate Resumes")
        uploaded_resumes = st.file_uploader(
            "Upload Candidate Resumes (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="resumes_uploader"
        )
        st.info("💡 Supports screening 10 to 50+ resumes simultaneously. Click '⚡ Load Demo Dataset' in the sidebar for instant testing!")

        if uploaded_resumes:
            st.success(f"📁 {len(uploaded_resumes)} resume file(s) selected for screening.")

    st.markdown("---")
    
    # Screening Action Trigger Button
    if st.button("🚀 Screen & Rank Candidates", type="primary", use_container_width=True):
        if not st.session_state.jd_text.strip():
            st.error("⚠️ Please upload or paste a Job Description before proceeding.")
        elif not uploaded_resumes and not st.session_state.screening_results:
            st.error("⚠️ Please upload at least one candidate resume or load the demo dataset.")
        else:
            with st.spinner("🔍 Processing documents, extracting skills, and calculating semantic embeddings..."):
                extractor = Extractor()
                similarity_engine = SimilarityEngine()
                scorer = CandidateScorer()
                recommender = CandidateRecommender()

                st.session_state.jd_info = extractor.extract_job_description_info(st.session_state.jd_text)

                results = []
                resume_texts = []
                parsed_resumes = []

                if uploaded_resumes:
                    for r_file in uploaded_resumes:
                        r_text = DocumentParser.parse_document(r_file, r_file.name)
                        if r_text:
                            r_info = extractor.extract_resume_info(r_text, r_file.name)
                            parsed_resumes.append(r_info)
                            resume_texts.append(r_text)

                    if resume_texts:
                        sem_scores = similarity_engine.compute_semantic_similarity(st.session_state.jd_text, resume_texts)
                        for idx, r_info in enumerate(parsed_resumes):
                            eval_res = scorer.evaluate_candidate(r_info, st.session_state.jd_info, sem_scores[idx])
                            report = recommender.generate_report(r_info, st.session_state.jd_info, eval_res)
                            results.append(report)

                        results.sort(key=lambda x: x["overall_score"], reverse=True)
                        st.session_state.screening_results = results
                        st.session_state.parsed_resumes = parsed_resumes
                        st.session_state.is_screened = True

                        export_to_csv(results)
                        export_to_json(results)
                        st.success("🎉 Candidates successfully screened and ranked!")


# ==========================================
# TAB 2: CANDIDATE RANKINGS
# ==========================================
with tab_rankings:
    st.subheader("Candidate Rankings Dashboard")

    if not st.session_state.is_screened or not st.session_state.screening_results:
        st.warning("⚠️ No candidates screened yet. Please upload files or click '⚡ Load Demo Dataset' in the sidebar.")
    else:
        results = st.session_state.screening_results

        # Top Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        total_cand = len(results)
        top_score = results[0]["overall_score"] if results else 0
        highly_rec_count = sum(1 for r in results if r["recommendation"] == "Highly Recommended")
        avg_score = round(sum(r["overall_score"] for r in results) / total_cand, 1) if total_cand else 0

        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{total_cand}</div><div class="metric-lbl">Total Candidates</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{top_score}/100</div><div class="metric-lbl">Top Candidate Score</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{highly_rec_count}</div><div class="metric-lbl">Highly Recommended</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_score}/100</div><div class="metric-lbl">Average Score</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Filters & Search
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
        with f_col1:
            category_filter = st.multiselect(
                "Filter by Recommendation",
                options=["Highly Recommended", "Recommended", "Consider", "Not Recommended"],
                default=["Highly Recommended", "Recommended", "Consider", "Not Recommended"]
            )
        with f_col2:
            search_query = st.text_input("🔍 Search Candidate Name or Skill", value="")
        with f_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            exp_csv_btn = st.download_button(
                "📥 Export CSV Results",
                data=pd.DataFrame([{
                    "Rank": i+1,
                    "Name": r["candidate_name"],
                    "Score": r["overall_score"],
                    "Recommendation": r["recommendation"],
                    "Skill Score": r["sub_scores"]["skill_score"],
                    "Exp Score": r["sub_scores"]["experience_score"],
                    "Edu Score": r["sub_scores"]["education_score"],
                    "Semantic Score": r["sub_scores"]["semantic_score"],
                } for i, r in enumerate(results)]).to_csv(index=False),
                file_name="ranked_candidates.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Filter Results
        filtered_results = []
        for rank, r in enumerate(results, 1):
            if r["recommendation"] in category_filter:
                if not search_query or search_query.lower() in r["candidate_name"].lower() or any(search_query.lower() in s.lower() for s in r["matching_skills"]):
                    filtered_results.append((rank, r))

        # Render Leaderboard Table
        st.markdown("### Candidate Leaderboard")
        
        for rank, r in filtered_results:
            rec_class = f"badge-{r['recommendation'].lower().replace(' ', '-')}"
            matched_tags = " ".join([f'<span class="skill-chip-match">{s}</span>' for s in r["matching_skills"][:5]])
            
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([0.6, 2.5, 1.2, 1.8, 2.5])
                with c1:
                    st.markdown(f"### #{rank}")
                with c2:
                    st.markdown(f"**{r['candidate_name']}**")
                    st.caption(f"📧 {r['email']} | 📞 {r['phone']}")
                with c3:
                    st.markdown(f"### **{r['overall_score']}** / 100")
                with c4:
                    st.markdown(f'<span class="{rec_class}">{r["recommendation"]}</span>', unsafe_allow_html=True)
                with c5:
                    st.markdown(f"**Matched Skills:**<br>{matched_tags}", unsafe_allow_html=True)
                
                st.divider()


# ==========================================
# TAB 3: CANDIDATE DETAILS
# ==========================================
with tab_details:
    st.subheader("Detailed Candidate Profile & Breakdown")

    if not st.session_state.is_screened or not st.session_state.screening_results:
        st.warning("⚠️ Please screen candidates first to view detailed profiles.")
    else:
        results = st.session_state.screening_results
        cand_names = [r["candidate_name"] for r in results]
        
        selected_cand_name = st.selectbox("Select Candidate to Evaluate", options=cand_names)
        cand_report = next((r for r in results if r["candidate_name"] == selected_cand_name), results[0])
        cand_info = next((p for p in st.session_state.get("parsed_resumes", []) if p.get("name") == selected_cand_name), {})

        col_left, col_right = st.columns([1.2, 1], gap="large")

        with col_left:
            # Candidate Overview Header Card
            st.markdown(f"## 👤 {cand_report['candidate_name']}")
            rec_color = RECOMMENDATION_COLORS.get(cand_report["recommendation"], "#3B82F6")
            st.markdown(f"### Status: <span style='color:{rec_color}; font-weight:bold;'>{cand_report['recommendation']}</span> (Score: {cand_report['overall_score']}/100)", unsafe_allow_html=True)

            st.markdown("#### Score Component Breakdown")
            sub = cand_report["sub_scores"]
            
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Skill Match (40%)", f"{sub['skill_score']}%")
            sc2.metric("Experience (30%)", f"{sub['experience_score']}%")
            sc3.metric("Education (15%)", f"{sub['education_score']}%")
            sc4.metric("Semantic (15%)", f"{sub['semantic_score']}%")

            st.markdown("---")

            # Skills Breakdown
            st.markdown("#### 🛠️ Skill Alignment")
            st.markdown("**Matched Skills:**")
            matched_html = "".join([f'<span class="skill-chip-match">{s}</span>' for s in cand_report["matching_skills"]]) or "None"
            st.markdown(matched_html, unsafe_allow_html=True)

            st.markdown("<br>**Missing Required Skills:**", unsafe_allow_html=True)
            missing_html = "".join([f'<span class="skill-chip-missing">{s}</span>' for s in cand_report["missing_skills"]]) or "None (Full skill coverage)"
            st.markdown(missing_html, unsafe_allow_html=True)

            st.markdown("---")

            # Strengths & Weaknesses
            st.markdown("#### ⚡ Strengths & Areas of Improvement")
            st.markdown("**Strengths:**")
            for st_item in cand_report["strengths"]:
                st.markdown(f"- ✅ {st_item}")

            st.markdown("**Areas of Improvement / Gaps:**")
            for wk_item in cand_report["weaknesses"]:
                st.markdown(f"- ⚠️ {wk_item}")

            st.markdown("---")
            st.markdown("#### 🎓 Background Summary")
            st.write(f"• **Experience:** {cand_report['experience_summary']}")
            st.write(f"• **Education:** {cand_report['education_summary']}")

        with col_right:
            st.markdown("### 🛠️ Candidate Evaluation Tools")

            llm_tab1, llm_tab2, llm_tab3, llm_tab4 = st.tabs(["📝 Score Summary", "❓ Interview Kit", "🎯 Gap Analysis", "📄 PDF Download"])

            recommender = CandidateRecommender()

            with llm_tab1:
                st.markdown("#### Executive Score Summary")
                explanation = recommender.explain_score(cand_report["candidate_name"], cand_report)
                st.info(explanation)

            with llm_tab2:
                st.markdown("#### Tailored Interview Kit")
                questions = recommender.generate_interview_questions(
                    cand_report["candidate_name"],
                    st.session_state.jd_info.get("job_title", "Position"),
                    cand_report["matching_skills"],
                    cand_report["missing_skills"]
                )
                for q in questions:
                    st.write(q)

            with llm_tab3:
                st.markdown("#### Actionable Skill Gap Analysis")
                gap_analysis = recommender.analyze_skill_gap(
                    cand_report["candidate_name"],
                    cand_report["missing_skills"],
                    st.session_state.jd_info.get("required_skills", [])
                )
                st.markdown(gap_analysis)

            with llm_tab4:
                st.markdown("#### Download PDF Evaluation Report")
                pdf_path = generate_candidate_pdf_report(cand_report)
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download {cand_report['candidate_name']} PDF Report",
                            data=f.read(),
                            file_name=f"{cand_report['candidate_name']}_Evaluation_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

            # Keyword Highlighted Resume Section
            st.markdown("---")
            st.markdown("### 🔍 Highlighted Resume Text")
            raw_text = cand_info.get("raw_text", "Resume text not available in memory.")
            if raw_text:
                highlighted = highlight_keywords_in_text(raw_text, cand_report["matching_skills"])
                st.markdown(f'<div style="max-height: 300px; overflow-y: scroll; padding: 1rem; border: 1px solid #E2E8F0; border-radius: 8px; font-family: monospace; font-size: 0.85rem; line-height: 1.6;">{highlighted}</div>', unsafe_allow_html=True)


# ==========================================
# TAB 4: ANALYTICS & INSIGHTS
# ==========================================
with tab_analytics:
    st.subheader("Recruitment Analytics & Insights")

    if not st.session_state.is_screened or not st.session_state.screening_results:
        st.warning("⚠️ Please screen candidates to view recruitment analytics.")
    else:
        results = st.session_state.screening_results
        df = pd.DataFrame([{
            "Candidate": r["candidate_name"],
            "Overall Score": r["overall_score"],
            "Recommendation": r["recommendation"],
            "Skill Score": r["sub_scores"]["skill_score"],
            "Experience Score": r["sub_scores"]["experience_score"],
            "Education Score": r["sub_scores"]["education_score"],
            "Semantic Score": r["sub_scores"]["semantic_score"],
        } for r in results])

        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("### Overall Candidate Ranking")
            fig_bar = px.bar(
                df,
                x="Overall Score",
                y="Candidate",
                orientation="h",
                color="Recommendation",
                color_discrete_map=RECOMMENDATION_COLORS,
                text="Overall Score",
                title="Candidates Ranked by Overall Score"
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=380)
            st.plotly_chart(fig_bar, use_container_width=True)

        with ch2:
            st.markdown("### Recommendation Distribution")
            rec_counts = df["Recommendation"].value_counts().reset_index()
            rec_counts.columns = ["Recommendation", "Count"]
            fig_pie = px.pie(
                rec_counts,
                names="Recommendation",
                values="Count",
                color="Recommendation",
                color_discrete_map=RECOMMENDATION_COLORS,
                hole=0.4,
                title="Candidate Tier Distribution"
            )
            fig_pie.update_layout(height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")

        ch3, ch4 = st.columns(2)

        with ch3:
            st.markdown("### Sub-Score Distribution Breakdown")
            df_sub = df.melt(id_vars=["Candidate"], value_vars=["Skill Score", "Experience Score", "Education Score", "Semantic Score"], var_name="Metric", value_name="Score")
            fig_group = px.bar(
                df_sub,
                x="Candidate",
                y="Score",
                color="Metric",
                barmode="group",
                title="Sub-Score Comparison Per Candidate"
            )
            fig_group.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_group, use_container_width=True)

        with ch4:
            st.markdown("### Skill Gap Analysis across Candidates")
            all_missing = []
            all_matched = []
            for r in results:
                all_missing.extend(r.get("missing_skills", []))
                all_matched.extend(r.get("matching_skills", []))

            if all_missing:
                missing_counts = pd.Series(all_missing).value_counts().reset_index()
                missing_counts.columns = ["Skill", "Frequency"]
                fig_miss = px.bar(
                    missing_counts.head(8),
                    x="Frequency",
                    y="Skill",
                    orientation="h",
                    color_discrete_sequence=["#EF4444"],
                    title="Most Frequently Missing Skills in Applicant Pool"
                )
                fig_miss.update_layout(yaxis=dict(autorange="reversed"), height=400)
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("🎉 All candidates cover required skills!")


# ==========================================
# TAB 5: RECRUITER ASSISTANT
# ==========================================
with tab_agent:
    st.subheader("💬 Recruiter Assistant")
    st.markdown("Ask natural language questions, shortlist applicants, schedule interviews, or save candidate records to your local database.")

    if not st.session_state.is_screened or not st.session_state.screening_results:
        st.warning("⚠️ Please screen candidates or click '⚡ Load Demo Dataset' in the sidebar to activate context.")
    else:
        agent = RecruiterAIAgent()

        st.markdown("##### ⚡ Quick Prompt Suggestions:")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        
        selected_prompt = ""
        with p_col1:
            if st.button("💡 Who is top candidate?", key="p1"):
                selected_prompt = "Who is the top candidate and why?"
        with p_col2:
            if st.button("📋 Shortlist Alex Rivera", key="p2"):
                selected_prompt = "Shortlist candidate Alex Rivera and save status to database."
        with p_col3:
            if st.button("📅 Schedule Sophia Chen", key="p3"):
                selected_prompt = "Schedule an interview with Sophia Chen for tomorrow at 10 AM."
        with p_col4:
            if st.button("📊 View Database Status", key="p4"):
                selected_prompt = "Show me all candidate status records in database."

        st.markdown("---")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Type your question or instruction...") or selected_prompt

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                agent_res = agent.process_query(
                    user_query=user_input,
                    candidates_data=st.session_state.screening_results,
                    jd_text=st.session_state.jd_text,
                    chat_history=st.session_state.messages
                )

                for act in agent_res.get("actions_taken", []):
                    st.success(act)

                reply = agent_res["response"]
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

        with st.expander("💾 Candidate Tracking Database Records (SQLite)"):
            db_records = agent.tool_get_database_status()
            if db_records:
                st.dataframe(pd.DataFrame(db_records), use_container_width=True)
            else:
                st.info("No saved candidate database records yet.")
