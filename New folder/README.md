# 🎯 AI Resume Screening Agent & REST API

A production-grade, high-performance AI Resume Screening & Candidate Ranking application built with Python 3.11+, Streamlit, FastAPI REST API, SentenceTransformers (`all-MiniLM-L6-v2`), Groq LLM API, and Plotly.

---

## 🌟 Key Features

- **Multi-Format Document Ingestion**: Parses PDF (pdfplumber + PyPDF2 fallback), Microsoft Word (.docx), and plain text (.txt) files.
- **Dual Interface**:
  - 🖥️ **Streamlit Web Application**: Interactive multi-tab recruiter dashboard (`http://localhost:8501`).
  - ⚡ **FastAPI REST API**: Production-grade HTTP REST server with Swagger UI (`http://localhost:8000/docs`).
- **NLP & Hybrid Extraction**: Extracts candidate contact details, skills, experience years, education, certifications, and projects using regex heuristics + Groq LLM structured JSON parsing.
- **Semantic Vector Matching**: Computes cosine similarity between Job Description requirements and candidate resumes using `SentenceTransformers` (`all-MiniLM-L6-v2`) with a robust TF-IDF fallback engine.
- **Weighted Candidate Scoring Formula**:
  $$\text{Overall Score} = 0.40 \cdot \text{Skill Match} + 0.30 \cdot \text{Experience Match} + 0.15 \cdot \text{Education Match} + 0.15 \cdot \text{Semantic Similarity}$$
- **Recommendation Tier Classification**: Categorizes applicants into *Highly Recommended*, *Recommended*, *Consider*, and *Not Recommended*.
- **LLM Feature Suite (Groq / OpenAI)**:
  - 🤖 **AI Score Explanation**: Executive natural-language summary explaining score derivation.
  - ❓ **AI Interview Question Generator**: Customized technical and behavioral interview kits targeting candidate strengths and missing skill gaps.
  - 🎯 **Skill Gap Analysis**: Actionable recommendations for onboarding or candidate upskilling.
  - 🔍 **Resume Keyword Highlighter**: Visual highlight of matched JD skills in raw resume text.
- **Data Export & PDF Reports**: Downloadable `ranked_candidates.csv`, `ranked_candidates.json`, and candidate-specific evaluation PDFs generated using ReportLab.
- **1-Click Demo Mode**: Built-in sample dataset to test the entire application out-of-the-box.

---

## 🌐 REST API Endpoints Overview

The REST API server runs at `http://localhost:8000` with automatic interactive Swagger documentation at `http://localhost:8000/docs`.

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `GET /` | `GET` | API status welcome landing page |
| `GET /health` | `GET` | Health check endpoint |
| `POST /api/screen-files` | `POST` | Screen and rank candidate resumes submitted as uploaded files (PDF, DOCX, TXT) against a JD file |
| `POST /api/screen-text` | `POST` | Screen and rank candidates submitted as raw text payload |
| `POST /api/extract-resume` | `POST` | Extract structured JSON fields (Name, Email, Phone, Skills, Edu, Exp) from a single resume file |
| `POST /api/extract-jd` | `POST` | Extract structured JSON requirements from a single JD file |

### Example cURL Request (`/api/screen-text`)
```bash
curl -X 'POST' \
  'http://localhost:8000/api/screen-text' \
  -H 'Content-Type: application/json' \
  -d '{
  "job_description": "Senior Python & AI Engineer with PyTorch, Docker, FastAPI and 5 years experience.",
  "resumes": [
    "Alex Rivera: Senior AI Engineer with 6 years experience in Python, PyTorch, Docker, FastAPI, LLMs."
  ]
}'
```

---

## ⚡ Quick Start & Running Applications

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables (`.env`)
```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_key_here
MODEL_PROVIDER=groq
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3. Run Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run FastAPI REST API Server
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Access interactive API docs at `http://localhost:8000/docs`.

---

## 📐 How Scoring Works

$$\text{Overall Score} = (0.40 \times \text{Skill Match}) + (0.30 \times \text{Experience Match}) + (0.15 \times \text{Education Match}) + (0.15 \times \text{Semantic Similarity})$$

- 🟢 **Highly Recommended**: Score $\ge 80.0$
- 🔵 **Recommended**: $65.0 \le \text{Score} < 80.0$
- 🟠 **Consider**: $50.0 \le \text{Score} < 65.0$
- 🔴 **Not Recommended**: Score $< 50.0$
