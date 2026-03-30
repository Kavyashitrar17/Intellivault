"""
rag_pipeline.py
---------------
IMPROVEMENTS OVER ORIGINAL:
  1. Uses get_vector_store() singleton — FAISS index loaded from disk only ONCE
     per process. Previously it was loaded 3 times per query call.
  2. Uses load_chunks() from save_chunks.py (JSON, not pickle) — one call,
     result passed directly to retrieve() so chunks aren't loaded twice.
  3. _maybe_sync_faiss_index only rebuilds if truly needed, and receives
     the already-loaded chunks + store to avoid a third disk read.
  4. Confidence thresholds come from config.py, not hardcoded magic numbers.
  5. Uses prompts.py message constants.
"""

import logging
from typing import Dict

from backend.retrieval.retriever import retrieve
from backend.retrieval.vector_store import get_vector_store
from backend.ingestion.save_chunks import load_chunks
from backend.ingestion.embedder import create_embeddings
from backend.llm.qa_chain import generate_answer
from backend.utils.prompts import NO_DOCUMENTS_MSG, NO_RESULTS_MSG
from backend.config import settings

logger = logging.getLogger(__name__)


def _sync_if_needed(chunks: list, store) -> None:
    """
    Rebuild the FAISS index only when its size diverges from chunks.json.
    Receives already-loaded chunks + store to avoid extra disk reads.
    """
    vec_count   = store.total_vectors
    chunk_count = len(chunks)

    if vec_count == chunk_count:
        return

    logger.warning(
        f"[Pipeline] FAISS/chunks mismatch: {vec_count} vectors vs {chunk_count} chunks. "
        "Rebuilding..."
    )
    try:
        embeddings = create_embeddings(chunks)
        if len(embeddings) == 0:
            logger.warning("[Pipeline] Rebuild skipped: no embeddings generated.")
            return
        store.rebuild_from_embeddings(embeddings)
    except Exception as e:
        logger.error(f"[Pipeline] Rebuild failed: {e}", exc_info=True)


def rag_pipeline(query: str) -> Dict:
    """
    Full RAG pipeline: load → sync check → retrieve → answer.

    Returns:
        {
            "answer":       str,
            "sources":      list[str],   # chunk text previews
            "source_files": list[str],   # filenames
            "confidence":   str          # "high" | "medium" | "low" | "none"
        }
    """
    logger.info(f"[Pipeline] Query: '{query}'")

    # --- Load chunks once ---
    chunks = load_chunks()
    if not chunks:
        return {
            "answer":       NO_DOCUMENTS_MSG,
            "sources":      [],
            "source_files": [],
            "confidence":   "none",
        }

    # --- Use singleton store (no extra disk read) ---
    store = get_vector_store()

    # --- Sync check (only rebuilds when sizes differ) ---
    _sync_if_needed(chunks, store)

    # --- Retrieve top chunks ---
    top_chunks = retrieve(query, chunks, k=settings.TOP_K, store=store)
    if not top_chunks:
        return {
            "answer":       NO_RESULTS_MSG,
            "sources":      [],
            "source_files": [],
            "confidence":   "low",
        }

    # --- Generate answer (real LLM or extractive fallback) ---
    answer = generate_answer(query, top_chunks)

    # --- Build response ---
    source_files    = list(set(c["source"] for c in top_chunks))
    source_previews = [
        c["text"][:150] + "..." if len(c["text"]) > 150 else c["text"]
        for c in top_chunks
    ]

    top_score  = float(top_chunks[0].get("score", 0) or 0)
    if top_score >= settings.CONFIDENCE_HIGH:
        confidence = "high"
    elif top_score >= settings.CONFIDENCE_MEDIUM:
        confidence = "medium"
    else:
        confidence = "low"

    logger.info(
        f"[Pipeline] Done | confidence={confidence} | "
        f"sources={source_files} | answer={answer[:80]}"
    )

    return {
        "answer":       answer,
        "sources":      source_previews,
        "source_files": source_files,
        "confidence":   confidence,
    }