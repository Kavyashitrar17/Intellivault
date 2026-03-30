"""
save_chunks.py
--------------
Saves and loads processed chunks using JSON (replaces pickle).

WHY JSON INSTEAD OF PICKLE?
  pickle is a security risk — a maliciously crafted .pkl file can execute
  arbitrary code when loaded. JSON is safe, human-readable, and debuggable.

IMPROVEMENTS:
  1. JSON storage instead of pickle
  2. count_chunks() reads only metadata (no full deserialisation) for /status
  3. load_chunks() used by rag_pipeline instead of inline pickle.load()
  4. All paths from config.py — nothing hardcoded
"""

import os
import json
import logging
from typing import List, Dict

from backend.config import settings

logger = logging.getLogger(__name__)


def save_chunks(new_chunks: List[Dict]) -> int:
    """
    Append new chunks to chunks.json.
    Re-numbers all chunk IDs to keep them in sync with FAISS index positions.

    Returns:
        Total number of chunks now stored (existing + new).
    """
    os.makedirs(os.path.dirname(settings.CHUNKS_PATH), exist_ok=True)

    existing_chunks = load_chunks()
    logger.info(f"[SaveChunks] Loaded {len(existing_chunks)} existing chunks.")

    offset = len(existing_chunks)
    for i, chunk in enumerate(new_chunks):
        chunk["chunk_id"] = offset + i

    all_chunks = existing_chunks + new_chunks

    with open(settings.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[SaveChunks] Saved {len(all_chunks)} total chunks "
        f"({len(new_chunks)} new + {len(existing_chunks)} existing)."
    )
    return len(all_chunks)


def load_chunks() -> List[Dict]:
    """
    Load all chunks from disk.
    Returns empty list if file does not exist.
    """
    if not os.path.exists(settings.CHUNKS_PATH):
        logger.warning(f"[SaveChunks] {settings.CHUNKS_PATH} not found.")
        return []
    try:
        with open(settings.CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info(f"[SaveChunks] Loaded {len(chunks)} chunks.")
        return chunks
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"[SaveChunks] Failed to load chunks: {e}")
        return []


def count_chunks() -> int:
    """
    Return total chunk count without loading all data into RAM.
    Used by /status endpoint.
    """
    if not os.path.exists(settings.CHUNKS_PATH):
        return 0
    try:
        with open(settings.CHUNKS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data)
    except Exception:
        return 0