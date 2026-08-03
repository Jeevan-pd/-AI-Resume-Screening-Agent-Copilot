"""
Flask Web Application for AI Resume Screening Agent.
Production-grade Flask server with full document ingestion, candidate ranking,
sub-score breakdown, CSV/JSON/PDF exports, analytics, and recruiter assistant.
"""

import os
import io
import json
import logging
from pathlib import Path
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, session
)

from config import (
    JOB_DESC_DIR, RESUMES_DIR, OUTPUT_DIR, RECOMMENDATION_COLORS
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

# Initialize Flask App
app = Flask(__name__)
app.secret_key = "super_secret_resume_screening_key"


# In-Memory Session / Global Store for current screening run
GLOBAL_STORE = {
    "jd_text": "",
    "jd_info": {},
    "screening_results": [],
    "parsed_resumes": [],
}


@app.route("/")
def index():
    """Ingestion & Setup Page."""
    return render_template("index.html")


@app.route("/demo-data", methods=["POST"])
def load_demo_data():
    """Load sample JD & Resumes dataset and screen immediately."""
    sample_jd_file = os.path.join(JOB_DESC_DIR, "Senior_AI_Engineer_JD.txt")
    if os.path.exists(sample_jd_file):
        GLOBAL_STORE["jd_text"] = DocumentParser.parse_document(sample_jd_file, "Senior_AI_Engineer_JD.txt")

    resumes_files = [os.path.join(RESUMES_DIR, f) for f in os.listdir(RESUMES_DIR)]
    
    extractor = Extractor()
    similarity_engine = SimilarityEngine()
    scorer = CandidateScorer()
    recommender = CandidateRecommender()

    GLOBAL_STORE["jd_info"] = extractor.extract_job_description_info(GLOBAL_STORE["jd_text"])

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

    if resume_texts and GLOBAL_STORE["jd_text"]:
        sem_scores = similarity_engine.compute_semantic_similarity(GLOBAL_STORE["jd_text"], resume_texts)
        for idx, r_info in enumerate(parsed_resumes):
            eval_res = scorer.evaluate_candidate(r_info, GLOBAL_STORE["jd_info"], sem_scores[idx])
            report = recommender.generate_report(r_info, GLOBAL_STORE["jd_info"], eval_res)
            results.append(report)

        results.sort(key=lambda x: x["overall_score"], reverse=True)
        GLOBAL_STORE["screening_results"] = results
        GLOBAL_STORE["parsed_resumes"] = parsed_resumes

        export_to_csv(results)
        export_to_json(results)
        flash("⚡ Demo Dataset loaded and screened successfully!", "success")

    return redirect(url_for("rankings"))


@app.route("/screen", methods=["POST"])
def screen_candidates():
    """Process uploaded Job Description and Candidate Resumes."""
    jd_file = request.files.get("jd_file")
    jd_text_input = request.form.get("jd_text", "").strip()
    resume_files = request.files.getlist("resume_files")

    jd_text = ""
    if jd_file and jd_file.filename:
        jd_bytes = jd_file.read()
        jd_text = DocumentParser.parse_document(jd_bytes, jd_file.filename)
    elif jd_text_input:
        jd_text = jd_text_input
    elif GLOBAL_STORE["jd_text"]:
        jd_text = GLOBAL_STORE["jd_text"]

    if not jd_text:
        flash("Please upload or paste a Job Description.", "error")
        return redirect(url_for("index"))

    GLOBAL_STORE["jd_text"] = jd_text

    extractor = Extractor()
    similarity_engine = SimilarityEngine()
    scorer = CandidateScorer()
    recommender = CandidateRecommender()

    GLOBAL_STORE["jd_info"] = extractor.extract_job_description_info(jd_text)

    results = []
    resume_texts = []
    parsed_resumes = []

    valid_resumes = [f for f in resume_files if f and f.filename]
    if valid_resumes:
        for r_file in valid_resumes:
            r_bytes = r_file.read()
            r_text = DocumentParser.parse_document(r_bytes, r_file.filename)
            if r_text:
                r_info = extractor.extract_resume_info(r_text, r_file.filename)
                parsed_resumes.append(r_info)
                resume_texts.append(r_text)

        if resume_texts:
            sem_scores = similarity_engine.compute_semantic_similarity(jd_text, resume_texts)
            for idx, r_info in enumerate(parsed_resumes):
                eval_res = scorer.evaluate_candidate(r_info, GLOBAL_STORE["jd_info"], sem_scores[idx])
                report = recommender.generate_report(r_info, GLOBAL_STORE["jd_info"], eval_res)
                results.append(report)

            results.sort(key=lambda x: x["overall_score"], reverse=True)
            GLOBAL_STORE["screening_results"] = results
            GLOBAL_STORE["parsed_resumes"] = parsed_resumes

            export_to_csv(results)
            export_to_json(results)
            flash(f"🎉 Successfully screened {len(results)} candidate resume(s)!", "success")
            return redirect(url_for("rankings"))
    
    # If no files uploaded but results exist
    if GLOBAL_STORE["screening_results"]:
        return redirect(url_for("rankings"))

    flash("Please upload at least one candidate resume or click 'Load Demo Data'.", "error")
    return redirect(url_for("index"))


@app.route("/rankings")
def rankings():
    """Candidate Leaderboard Page."""
    return render_template("rankings.html", results=GLOBAL_STORE["screening_results"])


@app.route("/candidate/<candidate_name>")
def candidate_detail(candidate_name):
    """Detailed Candidate Evaluation Page."""
    cand = next((r for r in GLOBAL_STORE["screening_results"] if r["candidate_name"].lower() == candidate_name.lower()), None)
    if not cand:
        flash(f"Candidate '{candidate_name}' not found.", "error")
        return redirect(url_for("rankings"))

    cand_info = next((p for p in GLOBAL_STORE["parsed_resumes"] if p.get("name", "").lower() == candidate_name.lower()), {})
    raw_text = cand_info.get("raw_text", "Resume text not loaded.")

    recommender = CandidateRecommender()
    explanation = recommender.explain_score(cand["candidate_name"], cand)
    questions = recommender.generate_interview_questions(
        cand["candidate_name"],
        GLOBAL_STORE["jd_info"].get("job_title", "Position"),
        cand["matching_skills"],
        cand["missing_skills"]
    )
    gap_analysis = recommender.analyze_skill_gap(
        cand["candidate_name"],
        cand["missing_skills"],
        GLOBAL_STORE["jd_info"].get("required_skills", [])
    )
    highlighted = highlight_keywords_in_text(raw_text, cand["matching_skills"])

    return render_template(
        "candidate_detail.html",
        candidate=cand,
        score_explanation=explanation,
        interview_questions=questions,
        gap_analysis=gap_analysis,
        highlighted_resume=highlighted
    )


@app.route("/download-pdf/<candidate_name>")
def download_pdf(candidate_name):
    """Download Candidate Evaluation Report PDF."""
    cand = next((r for r in GLOBAL_STORE["screening_results"] if r["candidate_name"].lower() == candidate_name.lower()), None)
    if not cand:
        flash("Candidate not found for PDF generation.", "error")
        return redirect(url_for("rankings"))

    pdf_path = generate_candidate_pdf_report(cand)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True, download_name=f"{candidate_name}_Evaluation_Report.pdf")
    
    flash("Failed to generate PDF report.", "error")
    return redirect(url_for("rankings"))


@app.route("/export-csv")
def export_csv():
    """Download Ranked Candidates CSV."""
    filepath = export_to_csv(GLOBAL_STORE["screening_results"])
    return send_file(filepath, as_attachment=True, download_name="ranked_candidates.csv")


@app.route("/export-json")
def export_json():
    """Download Ranked Candidates JSON."""
    filepath = export_to_json(GLOBAL_STORE["screening_results"])
    return send_file(filepath, as_attachment=True, download_name="ranked_candidates.json")


@app.route("/analytics")
def analytics():
    """Recruitment Analytics Dashboard Page."""
    results = GLOBAL_STORE["screening_results"]
    chart_candidates = [r["candidate_name"] for r in results]
    chart_scores = [r["overall_score"] for r in results]
    chart_tiers = [r["recommendation"] for r in results]

    return render_template(
        "analytics.html",
        results=results,
        chart_candidates=chart_candidates,
        chart_scores=chart_scores,
        chart_tiers=chart_tiers
    )


@app.route("/assistant")
def assistant_view():
    """Recruiter Assistant Page."""
    agent = RecruiterAIAgent()
    db_records = agent.tool_get_database_status()
    return render_template("assistant.html", db_records=db_records)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Recruiter Assistant Chat Endpoint."""
    data = request.get_json() or {}
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    agent = RecruiterAIAgent()
    res = agent.process_query(
        user_query=user_query,
        candidates_data=GLOBAL_STORE["screening_results"],
        jd_text=GLOBAL_STORE["jd_text"]
    )
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
