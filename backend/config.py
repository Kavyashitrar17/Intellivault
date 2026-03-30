"""
config.py
---------
Centralised settings for IntelliVault.

All tuneable constants and environment variables live here.
Import `settings` in any module — never hardcode paths or values.

Usage:
    from backend.config import settings
    print(settings.UPLOAD_FOLDER)

Requires:  pip install pydantic-settings python-dotenv
Create a .env file (already in .gitignore) to override defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    # -------------------------------------------------------
    # Paths
    # -------------------------------------------------------
    UPLOAD_FOLDER:  str = "data/uploads"
    CHUNKS_PATH:    str = "data/processed_chunks/chunks.json"
    INDEX_PATH:     str = "data/vector_db/faiss_index.index"

    # -------------------------------------------------------
    # Chunking
    # -------------------------------------------------------
    CHUNK_SIZE:   int = 400    # words per chunk
    CHUNK_OVERLAP: int = 50    # word overlap between chunks
    MIN_CHUNK_WORDS: int = 20  # discard chunks shorter than this

    # -------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------
    TOP_K:               int   = 5
    SEMANTIC_WEIGHT:     float = 0.7
    MIN_SEMANTIC_SCORE:  float = 0.25
    MIN_FINAL_SCORE:     float = 0.15

    # -------------------------------------------------------
    # QA / LLM
    # -------------------------------------------------------
    # Set LLM_PROVIDER to "flan" (local, free) or "openai" or "groq"
    LLM_PROVIDER:     str = "flan"
    OPENAI_API_KEY:   str = ""
    GROQ_API_KEY:     str = ""
    FLAN_MODEL_NAME:  str = "google/flan-t5-base"  # ~250 MB, runs on CPU

    # -------------------------------------------------------
    # API / Security
    # -------------------------------------------------------
    # Comma-separated allowed origins for CORS
    ALLOWED_ORIGINS: str = "http://localhost:8501"
    API_KEY:         str = ""   # Set this in .env to enable simple API key auth

    # -------------------------------------------------------
    # Confidence thresholds
    # -------------------------------------------------------
    CONFIDENCE_HIGH:   float = 0.6
    CONFIDENCE_MEDIUM: float = 0.35

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this everywhere
settings = Settings()

# Ensure required directories exist at startup
for path in [
    settings.UPLOAD_FOLDER,
    os.path.dirname(settings.CHUNKS_PATH),
    os.path.dirname(settings.INDEX_PATH),
]:
    os.makedirs(path, exist_ok=True)