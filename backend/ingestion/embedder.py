"""
embedder.py
-----------
IMPROVEMENTS OVER ORIGINAL:
  1. Lazy loading via get_model() — model only loads when first needed,
     not at import time. This means importing this module in tests or
     scripts no longer triggers a 90 MB download/load.
  2. Model name comes from config.py (settings.FLAN_MODEL_NAME is for QA;
     embedding model stays all-MiniLM-L6-v2 but is now configurable).
  3. create_embeddings() and embed_query() unchanged in behaviour.
"""

import logging
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# -------------------------------------------------------
# Lazy singleton
# -------------------------------------------------------
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Return the embedding model, loading it on first call only.
    Safe to call multiple times — returns cached instance.
    """
    global _model
    if _model is None:
        logger.info(f"[Embedder] Loading model '{EMBEDDING_MODEL_NAME}'...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("[Embedder] Model ready.")
    return _model


def create_embeddings(chunks: List[Dict], batch_size: int = 64) -> np.ndarray:
    """
    Generate L2-normalized embeddings for a list of chunk dicts.

    Returns:
        np.ndarray of shape (n, 384), dtype float32.
    """
    if not chunks:
        logger.warning("[Embedder] No chunks to embed — returning empty array.")
        return np.array([], dtype="float32")

    texts = [chunk["text"] for chunk in chunks]
    logger.info(f"[Embedder] Embedding {len(texts)} chunks...")

    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")

    # L2 normalize so FAISS inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    logger.info(f"[Embedder] Done. Shape: {embeddings.shape}")
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string. Returns shape (1, 384) float32.
    """
    model = get_model()
    embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding