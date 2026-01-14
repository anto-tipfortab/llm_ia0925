"""
Configuration module - loads environment variables and model settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Model parameters
MODEL_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "max_tokens": 1024,
    "top_p": 0.9,
}

# Paths
PDF_PATH = "data/TENERIFE.pdf"

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200