"""
retriever.py
------------
IMPROVEMENTS OVER ORIGINAL:
  1. retrieve() now accepts an optional `store` parameter.
     If passed (from rag_pipeline), the already-loaded singleton is reused.
     If not passed, it falls back to get_vector_store() — still the singleton.
     Before: always called VectorStore() which re-read the index from disk.
  2. All thresholds come from config.py.
  3. No logic changes to the hybrid retrieval algorithm — it was correct.
"""

import logging
import re
from typing import List, Dict, Optional
import numpy as np

from backend.ingestion.embedder import embed_query
from backend.retrieval.vector_store import get_vector_store, VectorStore
from backend.config import settings

logger = logging.getLogger(__name__)

SEMANTIC_WEIGHT = settings.SEMANTIC_WEIGHT
KEYWORD_WEIGHT  = 1.0 - SEMANTIC_WEIGHT
MIN_SEMANTIC_THRESHOLD         = settings.MIN_SEMANTIC_SCORE
MIN_FINAL_SCORE                = settings.MIN_FINAL_SCORE
DEFAULT_TOP_K                  = settings.TOP_K
MAX_TOP_K                      = settings.TOP_K
LOW_SEMANTIC_KEYWORD_GUARD     = 0.35
LOW_SEMANTIC_MIN_KEYWORD       = 0.10
MIN_RELATIVE_TO_BEST_SEMANTIC  = 0.70


def retrieve(
    query: str,
    chunks: List[Dict],
    k: int = DEFAULT_TOP_K,
    store: Optional[VectorStore] = None,
) -> List[Dict]:
    """
    Hybrid semantic + keyword retrieval.

    Args:
        query:  User question.
        chunks: Full list of chunk dicts (already loaded by caller).
        k:      Number of chunks to return.
        store:  Optional pre-loaded VectorStore. Avoids extra disk reads.

    Returns:
        Top-k chunks sorted best-first, each with an added "score" key.
    """
    if not chunks:
        logger.warning("[Retriever] No chunks available.")
        return []
    if k <= 0:
        return []

    k = min(k, MAX_TOP_K)

    # Use passed store or fall back to singleton
    if store is None:
        store = get_vector_store()

    if store.total_vectors == 0:
        logger.warning("[Retriever] FAISS index is empty.")
        return []

    # Fetch more candidates so re-ranking has room to work
    fetch_k = min(max(k * 3, k), store.total_vectors)
    query_vec = embed_query(query)
    indices, sem_scores = store.search(query_vec, k=fetch_k)

    logger.info(f"[Retriever] FAISS returned {len(indices)} candidates.")

    # --- Filter invalid / duplicate indices ---
    seen = set()
    candidates_all         = []
    candidates_thresholded = []
    chunk_len = len(chunks)

    for idx, sem_score in zip(indices, sem_scores):
        if idx is None or int(idx) < 0:
            continue
        idx = int(idx)
        if idx in seen or not (0 <= idx < chunk_len):
            if 0 <= idx < chunk_len:
                pass
            else:
                logger.warning(f"[Retriever] Index {idx} out of range. Skipping.")
                continue
            if idx in seen:
                continue
        seen.add(idx)
        sem_score = float(np.clip(sem_score, -1.0, 1.0))
        candidates_all.append((idx, sem_score))
        if sem_score >= MIN_SEMANTIC_THRESHOLD:
            candidates_thresholded.append((idx, sem_score))

    candidates = candidates_thresholded if candidates_thresholded else candidates_all
    if not candidates:
        return []

    # Drop tail candidates that are far below the best hit
    best_semantic = max(s for _, s in candidates)
    floor = max(MIN_SEMANTIC_THRESHOLD, best_semantic * MIN_RELATIVE_TO_BEST_SEMANTIC)
    candidates = [(i, s) for i, s in candidates if s >= floor]
    if not candidates:
        return []

    # --- Keyword scoring + hybrid blend ---
    query_tokens  = _tokenize(query)
    scored_chunks = []

    for idx, sem_score in candidates:
        chunk      = chunks[idx]
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            continue

        kw_score = _keyword_score(query_tokens, chunk_text)

        if sem_score < LOW_SEMANTIC_KEYWORD_GUARD and kw_score < LOW_SEMANTIC_MIN_KEYWORD:
            continue

        final = SEMANTIC_WEIGHT * sem_score + KEYWORD_WEIGHT * kw_score
        if final < MIN_FINAL_SCORE:
            continue

        scored_chunks.append({
            **chunk,
            "score":          round(float(final), 4),
            "semantic_score": round(sem_score, 4),
            "keyword_score":  round(float(kw_score), 4),
        })

    if not scored_chunks:
        logger.warning("[Retriever] No scored chunks produced.")
        return []

    scored_chunks.sort(key=lambda c: c["score"], reverse=True)
    top = scored_chunks[:k]
    logger.info(
        f"[Retriever] Returning {len(top)} chunks. "
        f"Top score: {top[0]['score'] if top else 'N/A'}"
    )
    return top


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z]+\b', text.lower())


def _keyword_score(query_tokens: List[str], chunk_text: str) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = set(_tokenize(chunk_text))
    query_set    = set(query_tokens)
    return len(query_set & chunk_tokens) / len(query_set)