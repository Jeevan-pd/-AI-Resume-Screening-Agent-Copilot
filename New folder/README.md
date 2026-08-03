# 🎯 AI Resume Screening Agent & REST API

A production-grade, high-performance AI Resume Screening & Candidate Ranking application built with Python 3.11+, Streamlit, FastAPI REST API, SentenceTransformers (`all-MiniLM-L6-v2`), Groq LLM API, and Plotly.

---

## 🚀 Deployment Guide (Streamlit vs. Vercel)

> [!IMPORTANT]
> **Why Streamlit apps fail on Vercel**:
> Vercel is a **Serverless Function platform** designed for stateless HTTP endpoints (like Next.js or FastAPI). Streamlit requires a persistent Python daemon process with WebSockets.
>
> - **To Deploy Streamlit Web App (Recommended)**: Use **Streamlit Community Cloud** (100% Free), **Render**, or **Railway**.
> - **To Deploy REST API on Vercel**: Use **Vercel** with the included `vercel.json` and `api.py`.

---

### Option A: Deploy Streamlit Web App on Streamlit Community Cloud (100% Free)

1. Push your repository to **GitHub**.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **"New App"**.
4. Select your Repository: `resume-screening-agent`.
5. Set Main file path: `app.py`.
6. Under **Advanced Settings -> Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```
7. Click **Deploy!** Your app will be live in 1-2 minutes.

---

### Option B: Deploy REST API Server on Vercel

1. Ensure `vercel.json` and `api.py` are in your root directory.
2. Install Vercel CLI: `npm i -g vercel`
3. Run deployment command:
   ```bash
   vercel
   ```
4. Set Environment Variable in Vercel Dashboard:
   `GROQ_API_KEY` = `your_groq_api_key`
5. Your REST API endpoints will be live at `https://your-project.vercel.app/docs`.

---

## 🌟 Key Features

- **Multi-Format Ingestion**: Parses PDF, DOCX, and TXT files.
- **Dual Interface**: Streamlit Web Dashboard (`app.py`) & FastAPI REST API (`api.py`).
- **Hybrid Extraction**: Regex taxonomy + Groq LLM structured JSON parsing.
- **Semantic Vector Matching**: `SentenceTransformers` (`all-MiniLM-L6-v2`) with TF-IDF fallback.
- **Weighted Scoring Formula**: $40\%$ Skill Match + $30\%$ Experience + $15\%$ Education + $15\%$ Semantic Similarity.
- **Interactive AI Recruiter Copilot**: Conversational AI assistant with SQLite candidate database tracking & tool execution.
