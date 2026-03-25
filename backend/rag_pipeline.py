"""
rag_pipeline.py
---------------
The main brain of IntelliVault. Orchestrates the full RAG pipeline.

FLOW:
  User query
    → embed query
    → FAISS semantic search (vector_store.py)
    → hybrid re-rank (retriever.py)
    → extractive answer (qa_chain.py)
    → formatted response

CHANGES FROM ORIGINAL:
  - Now uses retriever.py for hybrid retrieval (semantic + keyword)
    instead of raw FAISS search inline
  - Uses qa_chain.py for multi-sentence scoring instead of basic word matching
  - Response includes a "confidence" field so the UI can show relevance
  - Chunks are loaded ONCE per call, not twice
  - Handles all edge cases: empty index, no results, low confidence
"""

import os
import pickle
import logging
from typing import Dict

from backend.retrieval.retriever import retrieve
from backend.llm.qa_chain        import generate_answer
from backend.retrieval.vector_store import VectorStore
from backend.utils.prompts       import NO_DOCUMENTS_MSG, NO_RESULTS_MSG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CHUNKS_PATH = "data/processed_chunks/chunks.pkl"

def _maybe_sync_faiss_index(chunks: list) -> None:
    """
    Ensure FAISS index size matches the number of serialized chunks.

    If they diverge, FAISS can return indices that don't map to chunks.pkl
    (leading to empty results or guard-triggered skips).
    """
    if not chunks:
        return

    try:
        store = VectorStore()
        chunk_count = len(chunks)
        vec_count = store.total_vectors

        if vec_count == chunk_count:
            return

        logger.warning(
            "[Pipeline] FAISS/chunks mismatch detected. "
            f"FAISS vectors={vec_count} chunks={chunk_count}. Rebuilding FAISS index..."
        )

        # Rebuild FAISS deterministically from chunks.pkl.
        # This keeps retrieval index<->chunk mapping correct.
        from backend.ingestion.embedder import create_embeddings

        embeddings = create_embeddings(chunks)
        if len(embeddings) == 0:
            logger.warning("[Pipeline] Rebuild skipped: no embeddings generated.")
            return

        store.rebuild_from_embeddings(embeddings)
    except Exception as e:
        # Retrieval will still have guards, but we prefer correctness.
        logger.error(f"[Pipeline] Could not sync FAISS index: {e}", exc_info=True)


def load_chunks() -> list:
    """Load chunk list from disk. Returns empty list if file not found."""
    if not os.path.exists(CHUNKS_PATH):
        logger.warning("[Pipeline] chunks.pkl not found.")
        return []
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"[Pipeline] Loaded {len(chunks)} chunks.")
    return chunks


def rag_pipeline(query: str) -> Dict:
    """
    Full RAG pipeline: retrieve → rank → answer.

    Args:
        query: The user's natural language question.

    Returns:
        Dict with keys:
            answer       (str)  — the extracted answer
            sources      (list) — short previews of the top matching chunks
            source_files (list) — which documents the answer came from
            confidence   (str)  — "high" / "medium" / "low" based on top chunk score
    """
    logger.info(f"[Pipeline] Query: '{query}'")

    # --- Load chunks ---
    chunks = load_chunks()

    if not chunks:
        return {
            "answer":       NO_DOCUMENTS_MSG,
            "sources":      [],
            "source_files": [],
            "confidence":   "none",
        }

    # --- Retrieve top relevant chunks (hybrid semantic + keyword) ---
    _maybe_sync_faiss_index(chunks)
    top_chunks = retrieve(query, chunks, k=5)

    if not top_chunks:
        return {
            "answer":       NO_RESULTS_MSG,
            "sources":      [],
            "source_files": [],
            "confidence":   "low",
        }

    # --- Generate extractive answer ---
    answer = generate_answer(query, top_chunks)

    # --- Build response ---
    source_files   = list(set(c["source"] for c in top_chunks))
    source_previews = [
        c["text"][:150] + "..." if len(c["text"]) > 150 else c["text"]
        for c in top_chunks
    ]

    # Confidence based on top chunk's hybrid score
    top_score = float(top_chunks[0].get("score", 0) or 0)
    if top_score >= 0.6:
        confidence = "high"
    elif top_score >= 0.35:
        confidence = "medium"
    else:
        confidence = "low"

    logger.info(
        f"[Pipeline] Done. Confidence={confidence} | "
        f"Sources={source_files} | Answer: {answer[:80]}"
    )

    return {
        "answer":       answer,
        "sources":      source_previews,
        "source_files": source_files,
        "confidence":   confidence,
    }