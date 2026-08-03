"""
Vercel Serverless Entry Point for Flask Application.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app

# Vercel WSGI Handler
handler = app
