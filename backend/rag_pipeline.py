"""
rag_pipeline.py  (sources dedup fix)
--------------------------------------
FIX: Sources were showing 5 identical previews because all 5 FAISS hits
     pointed to the same chunk (document only had 1-2 unique chunks).
     Now deduplicates top_chunks by text similarity before building previews.
"""

import re
import logging
from typing import Dict, List

from backend.retrieval.retriever import retrieve
from backend.retrieval.vector_store import get_vector_store
from backend.ingestion.save_chunks import load_chunks
from backend.ingestion.embedder import create_embeddings
from backend.llm.qa_chain import generate_answer, score_confidence
from backend.utils.prompts import NO_DOCUMENTS_MSG, NO_RESULTS_MSG
from backend.config import settings

logger = logging.getLogger(__name__)


def _sync_if_needed(chunks: list, store) -> None:
    if store.total_vectors == len(chunks):
        return
    logger.warning(
        f"[Pipeline] Mismatch: {store.total_vectors} vectors vs {len(chunks)} chunks. Rebuilding..."
    )
    try:
        embeddings = create_embeddings(chunks)
        if len(embeddings):
            store.rebuild_from_embeddings(embeddings)
    except Exception as e:
        logger.error(f"[Pipeline] Rebuild failed: {e}", exc_info=True)


def _dedup_sources(chunks: List[Dict]) -> List[Dict]:
    """
    Remove near-duplicate chunks from the source list using text overlap.
    Prevents the sources field from showing 5 identical previews.
    """
    selected, seen_texts = [], []
    for chunk in chunks:
        text = chunk.get("text", "").strip()[:200]  # compare first 200 chars
        is_dup = any(
            _overlap(text, s) > 0.7
            for s in seen_texts
        )
        if not is_dup:
            selected.append(chunk)
            seen_texts.append(text)
    return selected


def _overlap(a: str, b: str) -> float:
    """Simple word-overlap ratio between two strings."""
    ta = set(re.findall(r"\b[a-z]+\b", a.lower()))
    tb = set(re.findall(r"\b[a-z]+\b", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def rag_pipeline(query: str) -> Dict:
    logger.info(f"[Pipeline] Query: '{query}'")

    chunks = load_chunks()
    if not chunks:
        return {"answer": NO_DOCUMENTS_MSG, "sources": [], "source_files": [], "confidence": "none"}

    store = get_vector_store()
    _sync_if_needed(chunks, store)

    top_chunks = retrieve(query, chunks, k=settings.TOP_K, store=store)
    if not top_chunks:
        return {"answer": NO_RESULTS_MSG, "sources": [], "source_files": [], "confidence": "low"}

    # Deduplicate before building sources (fixes identical preview issue)
    unique_chunks = _dedup_sources(top_chunks)

    answer = generate_answer(query, top_chunks)

    source_files    = list(set(c["source"] for c in unique_chunks))
    source_previews = [
        c["text"][:150] + "..." if len(c["text"]) > 150 else c["text"]
        for c in unique_chunks
    ]

    confidence = score_confidence(top_chunks, answer)

    logger.info(f"[Pipeline] Done | confidence={confidence} | answer={answer[:80]}")

    return {
        "answer":       answer,
        "sources":      source_previews,
        "source_files": source_files,
        "confidence":   confidence,
    }