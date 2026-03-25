"""
embedder.py
-----------
Converts text chunks into numerical vector embeddings.

WHAT ARE EMBEDDINGS?
  A vector embedding is a list of 384 numbers that represents
  the *meaning* of a sentence. Similar sentences have similar vectors.
  FAISS uses these vectors to find relevant chunks fast.

CHANGES FROM ORIGINAL:
- Model is loaded ONCE as a module-level singleton (not reloaded on every call)
- Embeddings are L2-normalized — this makes cosine similarity == dot product,
  which improves ranking quality with FAISS IndexFlatIP (if you switch later)
- Added batch_size param for large document sets (avoids memory spikes)
- Added logging to track how long embedding takes
"""

import logging
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Load model ONCE at module level.
# This means the model is loaded when the server starts,
# not on every /upload or /query request. Much faster.
# -------------------------------------------------------
logger.info("[Embedder] Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
_model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("[Embedder] Model ready.")


def create_embeddings(chunks: List[Dict], batch_size: int = 64) -> np.ndarray:
    """
    Generate normalized embeddings for a list of chunk dicts.

    Args:
        chunks:     List of chunk dicts with a "text" key.
        batch_size: How many chunks to embed at once.
                    Lower this if you run out of RAM on large documents.

    Returns:
        NumPy array of shape (num_chunks, 384), dtype float32.
        Each row is the embedding for the corresponding chunk.
    """
    if not chunks:
        logger.warning("[Embedder] No chunks to embed — returning empty array.")
        return np.array([]).astype("float32")

    texts = [chunk["text"] for chunk in chunks]

    logger.info(f"[Embedder] Embedding {len(texts)} chunks...")

    # encode() returns a numpy array by default
    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,  # set True if you want a progress bar in terminal
        convert_to_numpy=True,
    ).astype("float32")

    # --- L2 Normalization ---
    # After this, each vector has length 1.
    # This means FAISS L2 distance ≈ cosine distance, so ranking is more meaningful.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    embeddings = embeddings / norms

    logger.info(f"[Embedder] Done. Output shape: {embeddings.shape}")
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns shape (1, 384) float32 — ready for FAISS search.

    Kept separate from create_embeddings() so the RAG pipeline
    is clear about what's a document vs what's a query.
    """
    embedding = _model.encode([query], convert_to_numpy=True).astype("float32")

    # Normalize the query vector too (must match how documents are normalized)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding