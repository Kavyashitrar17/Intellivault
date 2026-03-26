"""
retriever.py
------------
Retrieves the most relevant chunks for a user query.

WHAT'S NEW (big upgrade from original):
  HYBRID RETRIEVAL = Semantic + Keyword scoring combined.

  Why?
    - Pure semantic (FAISS only) can miss exact keyword matches.
      E.g., query "what is deadlock" might rank a chunk about "mutex" higher
      than one that literally says "deadlock is..."
    - Pure keyword (BM25-style) misses paraphrases and synonyms.
    - Combining both gives you the best of both worlds.

HOW IT WORKS:
  1. FAISS gives us top-k candidates by semantic similarity (cosine score).
  2. We then compute a keyword overlap score for each candidate.
  3. Final score = α * semantic_score + (1-α) * keyword_score
  4. Re-rank by final score and return top results.

  α (alpha) controls the balance. Default 0.7 = mostly semantic.
"""

import logging
import re
from typing import List, Tuple, Dict
import numpy as np

from backend.ingestion.embedder       import embed_query
from backend.retrieval.vector_store   import VectorStore

logger = logging.getLogger(__name__)

# How much to weight semantic vs keyword score. 0.7 = 70% semantic, 30% keyword.
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT  = 1.0 - SEMANTIC_WEIGHT

# Minimum semantic score to consider a chunk (cosine scale: -1..1).
MIN_SEMANTIC_THRESHOLD = 0.25

# Minimum final hybrid score to keep in returned results.
MIN_FINAL_SCORE = 0.15

# Enforce concise result sets for downstream QA quality.
DEFAULT_TOP_K = 5
MAX_TOP_K = 5

# For weak semantic matches, require at least some keyword overlap.
LOW_SEMANTIC_KEYWORD_GUARD = 0.35
LOW_SEMANTIC_MIN_KEYWORD = 0.10

# Keep only candidates that are not too far from the best semantic hit.
# This prevents weak tail results from entering QA when fetch_k > k.
MIN_RELATIVE_TO_BEST_SEMANTIC = 0.70


def retrieve(query: str, chunks: List[Dict], k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Retrieve and re-rank the top-k most relevant chunks for a query.

    Args:
        query:  The user's question string.
        chunks: Full list of chunk dicts loaded from disk.
        k:      How many chunks to return.

    Returns:
        List of top chunks (dicts), sorted best-first.
        Each dict has: chunk_id, source, text, score (added here).
    """
    if not chunks:
        logger.warning("[Retriever] No chunks available to search.")
        return []

    if k <= 0:
        return []
    k = min(k, MAX_TOP_K)

    # --- Step 1: Semantic search via FAISS ---
    store = VectorStore()

    if store.total_vectors == 0:
        logger.warning("[Retriever] FAISS index is empty.")
        return []

    # Fetch more candidates than needed so re-ranking has room to work
    fetch_k = min(max(k * 3, k), store.total_vectors)
    query_vec = embed_query(query)
    indices, sem_scores = store.search(query_vec, k=fetch_k)

    logger.info(f"[Retriever] FAISS returned {len(indices)} candidates.")

    # --- Step 2: Filter invalid indices (index/chunk mismatch guard) ---
    seen = set()
    candidates_all = []
    candidates_thresholded = []

    chunk_len = len(chunks)
    for idx, sem_score in zip(indices, sem_scores):
        # FAISS may emit -1 indices in some edge cases.
        if idx is None or int(idx) < 0:
            continue
        idx = int(idx)
        if idx in seen:
            continue
        if not (0 <= idx < chunk_len):
            logger.warning(
                f"[Retriever] Index {idx} out of range (have {chunk_len} chunks). Skipping."
            )
            continue

        sem_score = float(np.clip(sem_score, -1.0, 1.0))
        seen.add(idx)
        candidates_all.append((idx, sem_score))
        if sem_score >= MIN_SEMANTIC_THRESHOLD:
            candidates_thresholded.append((idx, sem_score))

    # If thresholding wipes everything out, fall back to semantic top candidates.
    candidates = candidates_thresholded if candidates_thresholded else candidates_all
    if not candidates:
        logger.warning("[Retriever] No valid candidates after FAISS filtering.")
        return []

    # Secondary semantic gate: keep scores reasonably close to the best hit.
    # This improves precision by dropping weak tail chunks.
    best_semantic = max(score for _, score in candidates)
    relative_floor = max(MIN_SEMANTIC_THRESHOLD, best_semantic * MIN_RELATIVE_TO_BEST_SEMANTIC)
    candidates = [(idx, score) for idx, score in candidates if score >= relative_floor]
    if not candidates:
        logger.warning("[Retriever] No candidates passed relative semantic filter.")
        return []

    # --- Step 3: Keyword scoring ---
    query_tokens = _tokenize(query)

    scored_chunks = []
    for idx, sem_score in candidates:
        chunk = chunks[idx]
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            continue

        # IndexFlatIP + normalized vectors => cosine similarity.
        kw_score = _keyword_score(query_tokens, chunk_text)

        # Semantic-relatedness guard: if semantic signal is weak, keep only chunks
        # with at least minimal lexical support from the query terms.
        if sem_score < LOW_SEMANTIC_KEYWORD_GUARD and kw_score < LOW_SEMANTIC_MIN_KEYWORD:
            continue

        final = SEMANTIC_WEIGHT * sem_score + KEYWORD_WEIGHT * kw_score
        if final < MIN_FINAL_SCORE:
            continue

        scored_chunks.append({
            **chunk,
            "score": round(float(final), 4),
            "semantic_score": round(sem_score, 4),
            "keyword_score": round(float(kw_score), 4),
        })

    if not scored_chunks:
        logger.warning("[Retriever] No scored chunks produced.")
        return []

    # --- Step 4: Sort by final score, return top-k ---
    scored_chunks.sort(key=lambda c: c["score"], reverse=True)
    top_chunks = scored_chunks[: min(k, len(scored_chunks))]

    logger.info(
        f"[Retriever] Returning {len(top_chunks)} chunks. "
        f"Top score: {top_chunks[0]['score'] if top_chunks else 'N/A'}"
    )
    return top_chunks


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    Lowercase and split text into word tokens.
    Removes punctuation so 'deadlock.' matches 'deadlock'.
    """
    return re.findall(r'\b[a-z]+\b', text.lower())


def _keyword_score(query_tokens: List[str], chunk_text: str) -> float:
    """
    Simple keyword overlap score between 0.0 and 1.0.

    Formula:
        score = (# query words found in chunk) / (# unique query words)

    This is a simplified version of recall — how many query terms
    does this chunk cover?
    """
    if not query_tokens:
        return 0.0

    chunk_tokens = set(_tokenize(chunk_text))
    query_set    = set(query_tokens)

    matched = query_set & chunk_tokens
    return len(matched) / len(query_set)