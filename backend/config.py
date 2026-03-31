"""
config.py  (fixed)
------------------
Single Settings class — merges the two conflicting definitions.
Added model_config with extra="ignore" so unknown .env vars never crash startup.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import model_validator


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
    CHUNK_SIZE:      int = 300
    CHUNK_OVERLAP:   int = 50
    MIN_CHUNK_WORDS: int = 30

    # -------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------
    TOP_K:              int   = 5
    SEMANTIC_WEIGHT:    float = 0.7
    MIN_SEMANTIC_SCORE: float = 0.15
    MIN_FINAL_SCORE:    float = 0.10

    # -------------------------------------------------------
    # LLM
    # -------------------------------------------------------
    LLM_PROVIDER:    str = "flan"         # "flan" | "openai" | "groq"
    FLAN_MODEL_NAME: str = "google/flan-t5-base"
    OPENAI_API_KEY:  str = ""
    GROQ_API_KEY:    str = ""

    # -------------------------------------------------------
    # API / Security
    # -------------------------------------------------------
    ALLOWED_ORIGINS: str = "http://localhost:8501"
    API_KEY:         str = ""

    # -------------------------------------------------------
    # Confidence thresholds
    # -------------------------------------------------------
    CONFIDENCE_HIGH:   float = 0.6
    CONFIDENCE_MEDIUM: float = 0.35

    model_config = {
        "env_file":          ".env",
        "env_file_encoding": "utf-8",
        "extra":             "ignore",   # silently ignore unknown .env keys
    }


# Singleton
settings = Settings()

# Ensure required directories exist at startup
for _path in [
    settings.UPLOAD_FOLDER,
    os.path.dirname(settings.CHUNKS_PATH),
    os.path.dirname(settings.INDEX_PATH),
]:
    if _path:
        os.makedirs(_path, exist_ok=True)