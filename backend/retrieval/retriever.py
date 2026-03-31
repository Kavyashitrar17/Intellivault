"""
retriever.py  (robustness upgrade)
------------------------------------
FIXES:
  1. Lower thresholds — previous values were too strict for short/paraphrased queries
  2. Guaranteed fallback — if nothing passes threshold, top-k raw FAISS results
     are returned anyway so the LLM always has something to work with
  3. Query expansion — automatically expands common abbreviations and synonyms
     before embedding (e.g. "RAG" → adds "Retrieval Augmented Generation")
  4. Soft relative floor — replaced hard floor with a gentler one so tail
     results aren't cut unless they are genuinely unrelated
  5. Better logging — tells you exactly which stage filtered results out
"""

import logging
import re
from typing import List, Dict, Optional
import numpy as np

from backend.ingestion.embedder import embed_query
from backend.retrieval.vector_store import get_vector_store, VectorStore
from backend.config import settings

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Thresholds  (tuned down from previous overly-strict values)
# -------------------------------------------------------
SEMANTIC_WEIGHT   = settings.SEMANTIC_WEIGHT          # default 0.7
KEYWORD_WEIGHT    = 1.0 - SEMANTIC_WEIGHT

# Was 0.25 / 0.30 — lowered so paraphrased queries still retrieve chunks
MIN_SEMANTIC_SCORE = 0.15   # absolute floor on cosine similarity
MIN_FINAL_SCORE    = 0.10   # absolute floor on hybrid score

# How close to the best hit a result needs to be (was 0.70 — too strict)
MIN_RELATIVE_TO_BEST = 0.55

# Guard: weak semantic + weak keyword → skip  (relaxed)
LOW_SEM_GUARD      = 0.20
LOW_SEM_MIN_KW     = 0.05

DEFAULT_TOP_K      = settings.TOP_K
MAX_TOP_K          = 10  # allow a bit more headroom


# -------------------------------------------------------
# Query expansion map
# Expands abbreviations / domain terms before embedding.
# Add your own domain terms here.
# -------------------------------------------------------
_EXPANSION_MAP = {
    r"\brag\b":                   "RAG Retrieval Augmented Generation",
    r"\bllm\b":                   "LLM large language model",
    r"\bvector db\b":             "vector database FAISS embeddings",
    r"\bfaiss\b":                 "FAISS vector similarity search",
    r"\bqa\b":                    "QA question answering",
    r"\bnlp\b":                   "NLP natural language processing",
    r"\bml\b":                    "ML machine learning",
    r"\bai\b":                    "AI artificial intelligence",
    r"\bapi\b":                   "API application programming interface",
    r"\bos\b":                    "operating system",
    r"\bcpu\b":                   "CPU central processing unit",
    r"\bram\b":                   "RAM random access memory",
    r"\bhttp\b":                  "HTTP hypertext transfer protocol",
}


def _expand_query(query: str) -> str:
    """
    Expand abbreviations and synonyms in the query before embedding.

    Why: 'RAG' and 'Retrieval Augmented Generation' are semantically related
    but produce slightly different embedding vectors. Expanding the query
    increases the chance of a strong cosine match with document chunks.

    Example:
        "How does RAG work?" →
        "How does RAG Retrieval Augmented Generation work?"
    """
    expanded = query
    for pattern, replacement in _EXPANSION_MAP.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)

    if expanded != query:
        logger.info(f"[Retriever] Query expanded: '{query}' → '{expanded}'")

    return expanded


# -------------------------------------------------------
# Main retrieval function
# -------------------------------------------------------

def retrieve(
    query: str,
    chunks: List[Dict],
    k: int = DEFAULT_TOP_K,
    store: Optional[VectorStore] = None,
) -> List[Dict]:
    """
    Hybrid semantic + keyword retrieval with guaranteed fallback.

    Stages:
      1. Expand query (abbreviations / synonyms)
      2. FAISS semantic search → candidates
      3. Keyword scoring → hybrid score
      4. Threshold filtering
      5. FALLBACK: if nothing passes threshold, return top-k raw FAISS hits

    Args:
        query:  User question.
        chunks: Full list of chunk dicts (already loaded by caller).
        k:      Number of chunks to return.
        store:  Pre-loaded VectorStore singleton (avoids disk re-reads).

    Returns:
        Top-k chunks sorted best-first, each with added score fields.
        NEVER returns an empty list as long as chunks exist in the index.
    """
    if not chunks:
        logger.warning("[Retriever] No chunks available.")
        return []
    if k <= 0:
        return []

    k = min(k, MAX_TOP_K)
    store = store or get_vector_store()

    if store.total_vectors == 0:
        logger.warning("[Retriever] FAISS index is empty.")
        return []

    # --------------------------------------------------
    # Step 1: Embed expanded query
    # --------------------------------------------------
    expanded_query = _expand_query(query)
    query_vec      = embed_query(expanded_query)

    # Fetch more candidates than needed so re-ranking has room
    fetch_k  = min(max(k * 4, 20), store.total_vectors)
    indices, sem_scores = store.search(query_vec, k=fetch_k)
    logger.info(f"[Retriever] FAISS returned {len(indices)} raw candidates.")

    # --------------------------------------------------
    # Step 2: Validate indices
    # --------------------------------------------------
    seen        = set()
    valid_pairs = []   # (idx, sem_score)
    chunk_len   = len(chunks)

    for idx, sem_score in zip(indices, sem_scores):
        if idx is None or int(idx) < 0:
            continue
        idx = int(idx)
        if idx in seen:
            continue
        if not (0 <= idx < chunk_len):
            logger.warning(f"[Retriever] Index {idx} out of range ({chunk_len} chunks). Skipping.")
            continue
        seen.add(idx)
        sem_score = float(np.clip(sem_score, -1.0, 1.0))
        valid_pairs.append((idx, sem_score))

    if not valid_pairs:
        logger.warning("[Retriever] No valid candidates after index validation.")
        return []

    # --------------------------------------------------
    # Step 3: Hybrid scoring
    # --------------------------------------------------
    query_tokens  = _tokenize(query)        # keyword score uses original query
    scored_chunks = []
    fallback_pool = []   # collects ALL scored chunks for the fallback path

    best_semantic = max(s for _, s in valid_pairs)
    logger.info(f"[Retriever] Best semantic score: {best_semantic:.4f}")

    for idx, sem_score in valid_pairs:
        chunk      = chunks[idx]
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            continue

        kw_score = _keyword_score(query_tokens, chunk_text)
        final    = SEMANTIC_WEIGHT * sem_score + KEYWORD_WEIGHT * kw_score

        entry = {
            **chunk,
            "score":          round(float(final), 4),
            "semantic_score": round(sem_score, 4),
            "keyword_score":  round(float(kw_score), 4),
        }
        fallback_pool.append(entry)

        # Apply soft relative floor
        relative_floor = max(MIN_SEMANTIC_SCORE, best_semantic * MIN_RELATIVE_TO_BEST)

        passes_semantic  = sem_score   >= relative_floor
        passes_final     = final       >= MIN_FINAL_SCORE
        passes_kw_guard  = not (sem_score < LOW_SEM_GUARD and kw_score < LOW_SEM_MIN_KW)

        if passes_semantic and passes_final and passes_kw_guard:
            scored_chunks.append(entry)

    # --------------------------------------------------
    # Step 4: Fallback — never return empty
    # --------------------------------------------------
    if not scored_chunks:
        logger.warning(
            "[Retriever] No chunks passed threshold filters. "
            f"(best_sem={best_semantic:.4f}, MIN_SEMANTIC={MIN_SEMANTIC_SCORE}, "
            f"MIN_FINAL={MIN_FINAL_SCORE}). "
            "Falling back to top raw FAISS results."
        )
        fallback_pool.sort(key=lambda c: c["semantic_score"], reverse=True)
        top = fallback_pool[:k]
        for c in top:
            c["fallback"] = True   # flag so the pipeline can optionally warn the UI
        logger.info(f"[Retriever] Fallback returning {len(top)} chunks.")
        return top

    # --------------------------------------------------
    # Step 5: Sort and return top-k
    # --------------------------------------------------
    scored_chunks.sort(key=lambda c: c["score"], reverse=True)
    top = scored_chunks[:k]
    logger.info(
        f"[Retriever] Returning {len(top)} chunks. "
        f"Top score: {top[0]['score']} | Semantic: {top[0]['semantic_score']}"
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