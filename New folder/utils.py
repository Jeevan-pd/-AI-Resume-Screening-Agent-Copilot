"""
Utilities Module for AI Resume Screening Agent.
Provides CSV/JSON export routines, PDF evaluation report generation (ReportLab),
resume keyword highlighting, and demo dataset pre-population.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from config import OUTPUT_DIR, RESUMES_DIR, JOB_DESC_DIR, RECOMMENDATION_COLORS

logger = logging.getLogger(__name__)


# ==========================================
# EXPORT UTILITIES (CSV & JSON)
# ==========================================

def export_to_csv(results: List[Dict[str, Any]], filepath: str = None) -> str:
    """Export ranked candidate list to CSV."""
    if not filepath:
        filepath = os.path.join(OUTPUT_DIR, "ranked_candidates.csv")

    flattened = []
    for rank, r in enumerate(results, 1):
        flattened.append({
            "Rank": rank,
            "Candidate Name": r.get("candidate_name", "N/A"),
            "Email": r.get("email", "N/A"),
            "Phone": r.get("phone", "N/A"),
            "Overall Score": r.get("overall_score", 0.0),
            "Recommendation": r.get("recommendation", "N/A"),
            "Skill Score": r.get("sub_scores", {}).get("skill_score", 0.0),
            "Experience Score": r.get("sub_scores", {}).get("experience_score", 0.0),
            "Education Score": r.get("sub_scores", {}).get("education_score", 0.0),
            "Semantic Score": r.get("sub_scores", {}).get("semantic_score", 0.0),
            "Matching Skills": ", ".join(r.get("matching_skills", [])),
            "Missing Skills": ", ".join(r.get("missing_skills", [])),
        })

    df = pd.DataFrame(flattened)
    df.to_csv(filepath, index=False)
    logger.info(f"Exported candidate rankings to CSV: {filepath}")
    return filepath


def export_to_json(results: List[Dict[str, Any]], filepath: str = None) -> str:
    """Export detailed candidate reports to JSON."""
    if not filepath:
        filepath = os.path.join(OUTPUT_DIR, "ranked_candidates.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Exported candidate reports to JSON: {filepath}")
    return filepath


# ==========================================
# PDF REPORT GENERATOR (ReportLab)
# ==========================================

def generate_candidate_pdf_report(report_data: Dict[str, Any], output_path: str = None) -> str:
    """Generates a professional PDF Candidate Evaluation Report."""
    name = report_data.get("candidate_name", "Candidate")
    if not output_path:
        filename = f"report_{re.sub(r'[^a-zA-Z0-9]', '_', name)}.pdf"
        output_path = os.path.join(OUTPUT_DIR, filename)

    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=10
    )
    heading_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor("#334155"),
        leading=14
    )

    story = []

    # Title Banner
    story.append(Paragraph("AI RESUME EVALUATION REPORT", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3B82F6"), spaceAfter=15))

    # Candidate Summary Table
    rec_text = report_data.get("recommendation", "N/A")
    rec_color_hex = RECOMMENDATION_COLORS.get(rec_text, "#3B82F6")
    
    summary_data = [
        [Paragraph(f"<b>Candidate Name:</b> {name}", body_style), Paragraph(f"<b>Overall Score:</b> {report_data.get('overall_score', 0)}/100", body_style)],
        [Paragraph(f"<b>Email:</b> {report_data.get('email', 'N/A')}", body_style), Paragraph(f"<b>Recommendation:</b> <font color='{rec_color_hex}'><b>{rec_text}</b></font>", body_style)],
        [Paragraph(f"<b>Phone:</b> {report_data.get('phone', 'N/A')}", body_style), Paragraph("", body_style)],
    ]
    summary_table = Table(summary_data, colWidths=[270, 270])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))

    # Sub-Scores Breakdown Table
    story.append(Paragraph("Score Component Breakdown", heading_style))
    sub = report_data.get("sub_scores", {})
    sub_data = [
        ["Evaluation Metric", "Weight", "Sub-Score"],
        ["Skill Match", "40%", f"{sub.get('skill_score', 0)}%"],
        ["Experience Match", "30%", f"{sub.get('experience_score', 0)}%"],
        ["Education Match", "15%", f"{sub.get('education_score', 0)}%"],
        ["Semantic Similarity", "15%", f"{sub.get('semantic_score', 0)}%"],
    ]
    sub_table = Table(sub_data, colWidths=[240, 150, 150])
    sub_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(sub_table)
    story.append(Spacer(1, 15))

    # Matched & Missing Skills
    story.append(Paragraph("Skills Analysis", heading_style))
    matched_str = ", ".join(report_data.get("matching_skills", [])) or "None identified"
    missing_str = ", ".join(report_data.get("missing_skills", [])) or "None (Full coverage)"
    story.append(Paragraph(f"<b>Matched Skills:</b> {matched_str}", body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Missing Required Skills:</b> {missing_str}", body_style))
    story.append(Spacer(1, 12))

    # Strengths & Weaknesses
    story.append(Paragraph("Key Strengths & Areas of Improvement", heading_style))
    for st in report_data.get("strengths", []):
        story.append(Paragraph(f"• {st}", body_style))
    story.append(Spacer(1, 4))
    for wk in report_data.get("weaknesses", []):
        story.append(Paragraph(f"• {wk}", body_style))
    story.append(Spacer(1, 15))

    # Summaries
    story.append(Paragraph("Experience & Education Overview", heading_style))
    story.append(Paragraph(report_data.get("experience_summary", ""), body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(report_data.get("education_summary", ""), body_style))

    doc.build(story)
    return output_path


# ==========================================
# RESUME KEYWORD HIGHLIGHTER
# ==========================================

def highlight_keywords_in_text(text: str, keywords: List[str]) -> str:
    """Highlights matched keywords in text using HTML <mark> tags."""
    if not text or not keywords:
        return text

    # Sort keywords by length descending to prevent substring collisions
    sorted_kw = sorted(keywords, key=len, reverse=True)
    highlighted_text = text

    for kw in sorted_kw:
        if len(kw.strip()) < 2:
            continue
        pattern = re.compile(r'\b(' + re.escape(kw.strip()) + r')\b', re.IGNORECASE)
        highlighted_text = pattern.sub(
            r'<mark style="background-color: #FEF08A; color: #854D0E; font-weight: bold; padding: 2px 4px; border-radius: 4px;">\1</mark>',
            highlighted_text
        )

    return highlighted_text


# ==========================================
# DEMO DATA PRE-POPULATION
# ==========================================

def create_sample_dataset():
    """Populates realistic sample JDs and candidate resumes in data/ directory."""
    os.makedirs(JOB_DESC_DIR, exist_ok=True)
    os.makedirs(RESUMES_DIR, exist_ok=True)

    # 1. Sample Job Description
    sample_jd_path = os.path.join(JOB_DESC_DIR, "Senior_AI_Engineer_JD.txt")
    if not os.path.exists(sample_jd_path):
        jd_content = """
Job Title: Senior AI / Full Stack Engineer
Location: Remote / Hybrid

Role Overview:
We are seeking an experienced Senior AI Engineer to join our core AI Innovation team. You will lead the architectural design, development, and deployment of production LLM applications, retrieval-augmented generation (RAG) pipelines, and intelligent agentic workflows.

Required Technical Skills:
- Python, PyTorch, Scikit-Learn, Pandas, NumPy
- Large Language Models (LLM), Prompt Engineering, RAG, LangChain, LlamaIndex
- FastAPI, Flask, Docker, Kubernetes, CI/CD pipelines
- PostgreSQL, Redis, MongoDB, Vector Databases (Pinecone, ChromaDB)
- Cloud Platforms: AWS, Azure, or GCP
- System Architecture, RESTful APIs, Git, Microservices

Experience Required:
- 5+ years of software engineering and machine learning production experience.

Education Required:
- Bachelor's or Master's Degree in Computer Science, Artificial Intelligence, Data Science, or related engineering discipline.

Responsibilities:
- Build and scale production-grade generative AI applications and microservices.
- Optimize model inference latency, vector similarity search, and semantic indexing.
- Collaborate with cross-functional teams to integrate AI models into user-facing Streamlit/React web applications.
- Implement robust unit testing, CI/CD automation, and monitoring pipelines.
        """
        with open(sample_jd_path, "w", encoding="utf-8") as f:
            f.write(jd_content.strip())

    # 2. Sample Candidates
    candidates = [
        {
            "filename": "Alex_Rivera_Lead_AI_Architect.txt",
            "content": """
Alex Rivera
Email: alex.rivera@example.com | Phone: +1 (555) 234-5678 | San Francisco, CA

SUMMARY
Senior AI Architect with 6.5 years of industry experience specializing in LLM deployment, RAG architectures, and scalable Python microservices on AWS.

SKILLS
Python, PyTorch, TensorFlow, Scikit-Learn, Pandas, NumPy, LangChain, LlamaIndex, LLM, RAG, Prompt Engineering, FastAPI, Docker, Kubernetes, AWS, PostgreSQL, Redis, Pinecone, Git, Microservices, CI/CD.

EXPERIENCE
Lead AI Engineer | TechCorp Inc. (2021 - Present)
- Engineered scalable LLM RAG pipelines serving 500,000 daily active users using FastAPI, LangChain, and Pinecone vector database.
- Orchestrated Docker container deployments on Kubernetes EKS clusters, reducing model inference latency by 35%.

Machine Learning Engineer | DataVision LLC (2018 - 2021)
- Built automated computer vision and NLP classification models using PyTorch and Scikit-Learn.
- Streamlined data processing pipelines handling 10TB+ dataset using Pandas and SQL.

EDUCATION
Master of Science in Computer Science | Stanford University (2018)
Bachelor of Science in Computer Engineering | UC Berkeley (2016)

CERTIFICATIONS
AWS Certified Solutions Architect – Associate
DeepLearning.AI Generative AI & LLM Specialist
            """
        },
        {
            "filename": "Sophia_Chen_FullStack_ML_Developer.txt",
            "content": """
Sophia Chen
Email: sophia.chen@example.com | Phone: +1 (555) 345-6789 | Seattle, WA

PROFESSIONAL EXPERIENCE
Full Stack ML Developer | InnovateAI (2020 - Present) - 4 Years Exp
- Developed production web applications using Python, Streamlit, React, and FastAPI backends.
- Integrated OpenAI and HuggingFace models into business analytics dashboards.
- Utilized Docker, MongoDB, and PostgreSQL for backend cloud architecture on Azure.

SKILLS
Python, JavaScript, React, FastAPI, Flask, Scikit-Learn, Pandas, Streamlit, OpenAI, Docker, PostgreSQL, MongoDB, Git, Azure.

EDUCATION
Bachelor of Science in Data Science | University of Washington (2020)

PROJECTS
- AI Resume Screening Tool: Built interactive Streamlit dashboard computing semantic document similarity using SentenceTransformers.
            """
        },
        {
            "filename": "David_Miller_Backend_Software_Engineer.txt",
            "content": """
David Miller
Email: david.m@example.com | Phone: +1 (555) 456-7890 | Austin, TX

SUMMARY
Software Engineer with 3 years of backend experience building Python APIs and microservices.

EXPERIENCE
Backend Engineer | CloudScale Systems (2021 - Present)
- Developed RESTful APIs using Python, Django, and PostgreSQL.
- Managed Git repositories and CI/CD pipelines using GitHub Actions.

SKILLS
Python, Django, Flask, SQL, PostgreSQL, REST APIs, Git, Docker, Linux.

EDUCATION
Bachelor of Arts in Information Technology | University of Texas at Austin (2021)
            """
        },
        {
            "filename": "Emily_Watson_Junior_Data_Analyst.txt",
            "content": """
Emily Watson
Email: emily.watson@example.com | Phone: +1 (555) 567-8901 | Boston, MA

SUMMARY
Enthusiastic Junior Data Analyst with 1 year of experience creating business dashboards and statistical analysis.

SKILLS
Python, SQL, R, Pandas, Excel, Tableau, Communication, Problem Solving.

EXPERIENCE
Data Analyst Intern | Analytics Corp (2023 - Present)
- Created interactive dashboards in Tableau and Excel for executive reporting.
- Executed SQL queries to clean data and generate weekly metrics reports.

EDUCATION
Bachelor of Science in Mathematics | Boston University (2023)
            """
        }
    ]

    for cand in candidates:
        filepath = os.path.join(RESUMES_DIR, cand["filename"])
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(cand["content"].strip())
