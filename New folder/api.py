"""
FastAPI REST API Server for AI Resume Screening Agent.
Provides RESTful endpoints for automated resume parsing, candidate scoring, ranking, and interview kit generation.
"""

import os
import io
import logging
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import GROQ_API_KEY, OPENAI_API_KEY, MODEL_PROVIDER
from parser import DocumentParser
from extractor import Extractor
from similarity import SimilarityEngine
from scorer import CandidateScorer
from recommender import CandidateRecommender
from utils import export_to_csv, export_to_json, generate_candidate_pdf_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="AI Resume Screening Agent REST API",
    description="Production-grade API for automated candidate resume screening, NLP semantic matching, candidate ranking, and AI interview kit generation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Request Models
class TextScreeningRequest(BaseModel):
    job_description: str = Field(..., description="Job Description plain text")
    resumes: List[str] = Field(..., description="List of candidate resume plain texts")
    api_key: Optional[str] = Field(default="", description="Optional Groq/OpenAI API key override")


class CandidateScoreResponse(BaseModel):
    rank: int
    candidate_name: str
    email: str
    phone: str
    overall_score: float
    recommendation: str
    sub_scores: dict
    matching_skills: List[str]
    missing_skills: List[str]
    strengths: List[str]
    weaknesses: List[str]
    experience_summary: str
    education_summary: str


@app.get("/")
def read_root():
    """API Root Welcome Endpoint."""
    return {
        "service": "AI Resume Screening Agent REST API",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
def health_check():
    """Health check status endpoint."""
    return {"status": "healthy", "service": "AI Resume Screening API"}


@app.post("/api/screen-files", response_model=List[CandidateScoreResponse])
async def screen_resumes_files(
    jd_file: UploadFile = File(..., description="Job Description file (PDF, DOCX, TXT)"),
    resume_files: List[UploadFile] = File(..., description="Multiple Resume files (PDF, DOCX, TXT)"),
    api_key: Optional[str] = Form(default="", description="Optional API key override")
):
    """
    Screen and rank candidate resumes submitted as files against a Job Description file.
    Returns ranked list of candidate evaluation reports.
    """
    if not jd_file or not resume_files:
        raise HTTPException(status_code=400, detail="Job Description file and resume files are required.")

    # Read and parse JD
    jd_content = await jd_file.read()
    jd_text = DocumentParser.parse_document(jd_content, jd_file.filename)
    if not jd_text:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from Job Description file '{jd_file.filename}'.")

    active_key = api_key.strip() or GROQ_API_KEY or OPENAI_API_KEY
    extractor = Extractor(api_key=active_key)
    similarity_engine = SimilarityEngine()
    scorer = CandidateScorer()
    recommender = CandidateRecommender(api_key=active_key)

    jd_info = extractor.extract_job_description_info(jd_text)

    parsed_resumes = []
    resume_texts = []

    for r_file in resume_files:
        r_bytes = await r_file.read()
        r_text = DocumentParser.parse_document(r_bytes, r_file.filename)
        if r_text:
            r_info = extractor.extract_resume_info(r_text, r_file.filename)
            parsed_resumes.append(r_info)
            resume_texts.append(r_text)

    if not resume_texts:
        raise HTTPException(status_code=400, detail="Could not parse text from any provided resume file.")

    sem_scores = similarity_engine.compute_semantic_similarity(jd_text, resume_texts)
    results = []

    for idx, r_info in enumerate(parsed_resumes):
        eval_res = scorer.evaluate_candidate(r_info, jd_info, sem_scores[idx])
        report = recommender.generate_report(r_info, jd_info, eval_res)
        results.append(report)

    # Sort descending by score
    results.sort(key=lambda x: x["overall_score"], reverse=True)

    # Add rank field
    response_list = []
    for rank, r in enumerate(results, 1):
        r_dict = r.copy()
        r_dict["rank"] = rank
        response_list.append(r_dict)

    return response_list


@app.post("/api/screen-text", response_model=List[CandidateScoreResponse])
def screen_resumes_text(request: TextScreeningRequest):
    """
    Screen and rank candidate resumes submitted as raw text against a Job Description text.
    """
    if not request.job_description.strip() or not request.resumes:
        raise HTTPException(status_code=400, detail="Job description text and resume texts are required.")

    active_key = request.api_key.strip() or GROQ_API_KEY or OPENAI_API_KEY
    extractor = Extractor(api_key=active_key)
    similarity_engine = SimilarityEngine()
    scorer = CandidateScorer()
    recommender = CandidateRecommender(api_key=active_key)

    jd_info = extractor.extract_job_description_info(request.job_description)

    parsed_resumes = []
    resume_texts = []

    for idx, r_text in enumerate(request.resumes):
        if r_text.strip():
            r_info = extractor.extract_resume_info(r_text, f"Candidate_{idx+1}.txt")
            parsed_resumes.append(r_info)
            resume_texts.append(r_text)

    if not resume_texts:
        raise HTTPException(status_code=400, detail="No valid text provided in resumes.")

    sem_scores = similarity_engine.compute_semantic_similarity(request.job_description, resume_texts)
    results = []

    for idx, r_info in enumerate(parsed_resumes):
        eval_res = scorer.evaluate_candidate(r_info, jd_info, sem_scores[idx])
        report = recommender.generate_report(r_info, jd_info, eval_res)
        results.append(report)

    results.sort(key=lambda x: x["overall_score"], reverse=True)

    response_list = []
    for rank, r in enumerate(results, 1):
        r_dict = r.copy()
        r_dict["rank"] = rank
        response_list.append(r_dict)

    return response_list


@app.post("/api/extract-resume")
async def extract_resume_file(file: UploadFile = File(...)):
    """Extract structured candidate information (Name, Email, Phone, Skills, Education, Experience) from a single resume file."""
    content = await file.read()
    text = DocumentParser.parse_document(content, file.filename)
    if not text:
        raise HTTPException(status_code=400, detail="Failed to parse document text.")
    
    extractor = Extractor()
    extracted_data = extractor.extract_resume_info(text, file.filename)
    return extracted_data


@app.post("/api/extract-jd")
async def extract_jd_file(file: UploadFile = File(...)):
    """Extract structured Job Description details (Required Skills, Min Experience, Education) from a single JD file."""
    content = await file.read()
    text = DocumentParser.parse_document(content, file.filename)
    if not text:
        raise HTTPException(status_code=400, detail="Failed to parse document text.")
    
    extractor = Extractor()
    extracted_data = extractor.extract_job_description_info(text)
    return extracted_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
