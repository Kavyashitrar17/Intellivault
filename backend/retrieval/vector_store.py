"""
vector_store.py
---------------
Thin wrapper around a FAISS index.

WHY A WRAPPER?
  It keeps FAISS details in one place. If you switch from FAISS to
  another vector DB later, only this file needs to change.

CHANGES FROM ORIGINAL:
- Switched from IndexFlatL2 → IndexFlatIP (Inner Product).
  With normalized embeddings (from embedder.py), inner product == cosine similarity.
  Cosine similarity ranks semantic matches better than raw L2 distance.
- Added save() and load() methods — used by api.py
- Added a scores_and_indices() method that returns BOTH distances and indices
  so the caller can filter by confidence score
"""

import os
import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)

DEFAULT_DIMENSION = 384   # all-MiniLM-L6-v2 outputs 384-dim vectors
INDEX_PATH = "data/vector_db/faiss_index.index"


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors row-wise (safe for zero vectors)."""
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class VectorStore:
    """
    Wraps a FAISS index with save/load and search functionality.

    Usage:
        store = VectorStore()
        store.add(embeddings_np)
        indices, scores = store.search(query_embedding, k=5)
        store.save()
    """

    def __init__(self, index_path: str = INDEX_PATH, dimension: int = DEFAULT_DIMENSION):
        # Backwards-compat:
        # Some scripts used `VectorStore(384)` (where 384 was intended as dimension).
        # If the first positional argument is an int, treat it as `dimension`.
        if isinstance(index_path, (int, np.integer)) and dimension == DEFAULT_DIMENSION:
            dimension = int(index_path)
            index_path = INDEX_PATH

        self.index_path = index_path
        self.dimension = int(dimension)
        self.index = self._load_or_create()

    def _load_or_create(self) -> faiss.Index:
        """Load index from disk if it exists, otherwise create a fresh one."""
        if os.path.exists(self.index_path):
            index = faiss.read_index(self.index_path)
            logger.info(
                f"[VectorStore] Loaded index from {self.index_path} "
                f"({index.ntotal} vectors)"
            )
            return index
        else:
            logger.warning(
                "[VectorStore] No index found — creating new IndexFlatIP."
            )
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            # IndexFlatIP = brute-force inner product (cosine with normalized vecs)
            return faiss.IndexFlatIP(self.dimension)

    def add(self, embeddings: np.ndarray):
        """
        Add embeddings to the index.
        embeddings must be shape (n, dimension) and float32.
        """
        if embeddings is None or len(embeddings) == 0:
            logger.warning("[VectorStore] No embeddings provided to add(). Skipping.")
            return

        embeddings = np.asarray(embeddings)
        embeddings = embeddings.astype("float32", copy=False)

        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings with shape (n, {self.dimension}), got {embeddings.shape}")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings dimension {self.dimension}, got {embeddings.shape[1]}"
            )

        # Defensive normalization keeps cosine behavior correct even if
        # upstream ingestion changes and sends non-normalized vectors.
        embeddings = _l2_normalize(embeddings)
        self.index.add(embeddings)
        logger.info(
            f"[VectorStore] Added {len(embeddings)} vectors. "
            f"Total: {self.index.ntotal}"
        )

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Search for the k most similar vectors.

        Args:
            query_embedding: Shape (1, 384), float32, normalized.
            k:               Number of results to return.

        Returns:
            indices: list of int positions in the chunk list
            scores:  list of float similarity scores (higher = more relevant)
        """
        if self.index.ntotal == 0:
            logger.warning("[VectorStore] Index is empty. Nothing to search.")
            return [], []

        if k <= 0:
            return [], []

        query_embedding = np.asarray(query_embedding).astype("float32", copy=False)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, got {query_embedding.shape[1]}"
            )

        # Defensive normalization: with IndexFlatIP this makes scores cosine similarities.
        query_embedding = _l2_normalize(query_embedding)

        k = min(int(k), self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        # FAISS can technically return -1 indices in some edge cases.
        indices_list = indices[0].tolist()
        scores_list = scores[0].tolist()
        return indices_list, scores_list

    def scores_and_indices(self, query_embedding: np.ndarray, k: int = 5):
        """
        Backwards-compatible helper: returns (indices, scores).
        """
        indices, scores = self.search(query_embedding, k=k)
        return indices, scores

    def add_embeddings(self, embeddings: np.ndarray):
        """
        Backwards-compatible alias for older scripts.
        """
        return self.add(embeddings)

    def save(self):
        """Persist the current index to disk."""
        faiss.write_index(self.index, self.index_path)
        logger.info(f"[VectorStore] Saved index to {self.index_path}")

    def rebuild_from_embeddings(self, embeddings: np.ndarray):
        """
        Replace the current FAISS index with embeddings (rebuild from scratch).
        """
        embeddings = np.asarray(embeddings)
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Cannot rebuild FAISS index with empty embeddings.")

        embeddings = embeddings.astype("float32", copy=False)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings with shape (n, d), got {embeddings.shape}")

        embeddings = _l2_normalize(embeddings)

        self.dimension = int(embeddings.shape[1])
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        logger.info(f"[VectorStore] Rebuilt index with {self.index.ntotal} vectors.")
        self.save()

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal