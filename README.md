# 🎯 AI Resume Screening Agent & REST API

A simple, production-grade, 100% self-contained AI Resume Screening & Candidate Ranking application built with Python 3.11+, Streamlit, FastAPI REST API, SentenceTransformers (`all-MiniLM-L6-v2`), and Plotly.

---

## ⚡ 100% Self-Contained Architecture (Zero API Keys Required!)

This application runs **completely offline & key-free**:
- **No Groq / OpenAI / External API keys required**.
- **Local Semantic Embeddings**: Powered by `sentence-transformers/all-MiniLM-L6-v2` with a robust TF-IDF fallback.
- **Fast NLP Information Extraction**: Rule-based taxonomy matching for technical skills, experience years, education levels, and contact info.
- **Recruiter Assistant & SQLite Database**: Automated candidate shortlisting, interview scheduling, and status tracking persisted in local SQLite DB.

---

## 🚀 Deployment Guide

### Option 1: Streamlit Community Cloud (100% Free & Recommended for UI)

1. Push your repository to **GitHub**.
2. Go to **[share.streamlit.io](https://share.streamlit.io)**.
3. Click **New App**, select your repository, and set main file path to `app.py`.
4. Click **Deploy!** (No API secret configuration needed).

---

### Option 2: Vercel REST API Server (`api_app.py` / `api/index.py`)

1. Push repository to **GitHub**.
2. Import project in **[vercel.com](https://vercel.com)**.
3. Click **Deploy**. Vercel will automatically route all API requests to `api/index.py`.

---

## 🌟 Key Features

- **Multi-Format Document Ingestion**: Parses PDF, DOCX, and TXT files.
- **Dual Interface**:
  - 🖥️ **Streamlit Web App**: Multi-tab recruiter dashboard (`app.py`).
  - ⚡ **FastAPI REST API**: OpenAPI Swagger documentation (`api_app.py` / `api/index.py`).
- **Weighted Candidate Scoring Formula**:
  $$\text{Overall Score} = 0.40 \cdot \text{Skill Match} + 0.30 \cdot \text{Experience Match} + 0.15 \cdot \text{Education Match} + 0.15 \cdot \text{Semantic Similarity}$$
- **Recommendation Categories**: Highly Recommended ($\ge 80$), Recommended ($65-79$), Consider ($50-64$), Not Recommended ($<50$).
- **Data Export & PDF Reports**: Downloadable CSV, JSON, and ReportLab PDF evaluation reports.
