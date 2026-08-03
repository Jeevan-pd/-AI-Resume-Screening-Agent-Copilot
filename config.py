"""
Configuration file for AI Resume Screening Agent.
Contains application constants, scoring weights, paths, taxonomy, and theme settings.
"""

from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESUMES_DIR = DATA_DIR / "resumes"
JOB_DESC_DIR = DATA_DIR / "job_description"
OUTPUT_DIR = BASE_DIR / "output"
ASSETS_DIR = BASE_DIR / "assets"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

# Ensure essential directories exist
for path in [DATA_DIR, RESUMES_DIR, JOB_DESC_DIR, OUTPUT_DIR, ASSETS_DIR, SCREENSHOTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Embedding Model Name
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Scoring Formula Weights (Must sum to 1.0)
WEIGHTS = {
    "skill_match": 0.40,
    "experience_match": 0.30,
    "education_match": 0.15,
    "semantic_similarity": 0.15,
}

# Recommendation Category Thresholds (Score out of 100)
RECOMMENDATION_THRESHOLDS = {
    "Highly Recommended": 80.0,
    "Recommended": 65.0,
    "Consider": 50.0,
    "Not Recommended": 0.0,
}

# Recommendation Color Codes
RECOMMENDATION_COLORS = {
    "Highly Recommended": "#10B981",  # Vibrant Emerald
    "Recommended": "#3B82F6",         # Electric Blue
    "Consider": "#F59E0B",            # Amber
    "Not Recommended": "#EF4444",      # Coral Red
}

# Education Levels Priority Map
EDUCATION_HIERARCHY = {
    "phd": 5,
    "doctorate": 5,
    "master": 4,
    "masters": 4,
    "m.s.": 4,
    "m.tech": 4,
    "bachelor": 3,
    "bachelors": 3,
    "b.s.": 3,
    "b.tech": 3,
    "b.e.": 3,
    "associate": 2,
    "diploma": 2,
    "high school": 1,
}

# Common Technical & Soft Skill Taxonomy for Keyword Extraction
COMMON_SKILLS_TAXONOMY = set([
    # Programming Languages
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "golang", "rust", "ruby", "php", "sql", "r", "scala", "swift", "kotlin", "html", "css", "bash", "shell",
    # Frameworks & Libraries
    "react", "react.js", "next.js", "vue", "angular", "node.js", "express", "django", "flask", "fastapi", "spring boot", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "streamlit", "opencv", "nltk", "spacy", "langchain", "llama-index", "huggingface", "tailwind", "bootstrap",
    # AI & ML Concepts
    "machine learning", "deep learning", "nlp", "natural language processing", "computer vision", "llm", "large language models", "generative ai", "rag", "retrieval augmented generation", "neural networks", "transformers", "feature engineering", "model deployment", "fine-tuning", "bert", "prompt engineering",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform", "ci/cd", "git", "github", "gitlab", "linux", "nginx", "microservices", "serverless", "jenkins", "ansible",
    # Databases & Data
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite", "snowflake", "bigquery", "pinecone", "chromadb", "weaviate", "spark", "hadoop", "etl", "data warehousing",
    # Soft Skills & Management
    "leadership", "project management", "agile", "scrum", "communication", "problem solving", "teamwork", "critical thinking", "collaboration", "analytical skills"
])
