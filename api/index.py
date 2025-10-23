"""
Vercel serverless function entry point for FastAPI
"""
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path so we can import from babycare
sys.path.append(str(Path(__file__).parent.parent))

from main import app

# Vercel expects this to be the default export
handler = app
