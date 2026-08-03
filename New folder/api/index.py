"""
Vercel Serverless Entry Point for FastAPI Application.
Routes all Vercel HTTP requests to the main FastAPI app.
"""

import sys
from pathlib import Path

# Add project root directory to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api_app import app

# Vercel Serverless Handler
handler = app
