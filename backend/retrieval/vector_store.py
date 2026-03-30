"""
vector_store.py
---------------
FAISS index wrapper with a module-level singleton.

IMPROVEMENTS OVER ORIGINAL:
  1. Singleton via get_vector_store() — the index is loaded from disk ONCE
     per process. Every subsequent call returns the same object.
     Before: every VectorStore() call re-read the .index file from disk.
     After:  one disk read at startup, zero disk reads per query.
  2. reset_vector_store() for /reset endpoint to clear the singleton.
  3. No logic changes to FAISS operations — those were already correct.
"""

import os
import logging
import numpy as np
import faiss

from backend.config import settings

logger = logging.getLogger(__name__)

DEFAULT_DIMENSION = 384  # all-MiniLM-L6-v2

# -------------------------------------------------------
# Module-level singleton
# -------------------------------------------------------
_store_instance: "VectorStore | None" = None


def get_vector_store() -> "VectorStore":
    """
    Return the shared VectorStore instance, creating it on first call.
    Subsequent calls return the same object — no disk read.
    """
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore()
    return _store_instance


def reset_vector_store():
    """Discard the singleton (used by /reset endpoint)."""
    global _store_instance
    _store_instance = None
    logger.info("[VectorStore] Singleton cleared.")


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


# -------------------------------------------------------
# VectorStore class
# -------------------------------------------------------
class VectorStore:
    """
    Thin wrapper around a FAISS IndexFlatIP index.

    Use get_vector_store() instead of instantiating directly.
    """

    def __init__(self,
                 index_path: str = None,
                 dimension: int = DEFAULT_DIMENSION):
        # Backwards-compat: VectorStore(384) treated as dimension
        if isinstance(index_path, (int, np.integer)):
            dimension  = int(index_path)
            index_path = None

        self.index_path = index_path or settings.INDEX_PATH
        self.dimension  = int(dimension)
        self.index      = self._load_or_create()

    def _load_or_create(self) -> faiss.Index:
        if os.path.exists(self.index_path):
            index = faiss.read_index(self.index_path)
            logger.info(
                f"[VectorStore] Loaded {self.index_path} ({index.ntotal} vectors)"
            )
            return index
        logger.warning("[VectorStore] No index found — creating new IndexFlatIP.")
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        return faiss.IndexFlatIP(self.dimension)

    def add(self, embeddings: np.ndarray):
        if embeddings is None or len(embeddings) == 0:
            logger.warning("[VectorStore] add() called with empty embeddings.")
            return
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError(f"Expected (n, {self.dimension}), got {embeddings.shape}")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")
        embeddings = _l2_normalize(embeddings)
        self.index.add(embeddings)
        logger.info(f"[VectorStore] Added {len(embeddings)} vectors. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Returns (indices, scores) — higher score = more relevant.
        """
        if self.index.ntotal == 0:
            return [], []
        if k <= 0:
            return [], []

        q = np.asarray(query_embedding, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError(f"Query dimension mismatch: expected {self.dimension}, got {q.shape[1]}")

        q = _l2_normalize(q)
        k = min(int(k), self.index.ntotal)
        scores, indices = self.index.search(q, k)
        return indices[0].tolist(), scores[0].tolist()

    def save(self):
        faiss.write_index(self.index, self.index_path)
        logger.info(f"[VectorStore] Saved to {self.index_path}")

    def rebuild_from_embeddings(self, embeddings: np.ndarray):
        embeddings = np.asarray(embeddings, dtype="float32")
        if len(embeddings) == 0:
            raise ValueError("Cannot rebuild with empty embeddings.")
        embeddings = _l2_normalize(embeddings)
        self.dimension = int(embeddings.shape[1])
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        logger.info(f"[VectorStore] Rebuilt with {self.index.ntotal} vectors.")
        self.save()

    # Backward-compat aliases
    def add_embeddings(self, embeddings):
        return self.add(embeddings)

    def scores_and_indices(self, query_embedding, k=5):
        return self.search(query_embedding, k=k)

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal