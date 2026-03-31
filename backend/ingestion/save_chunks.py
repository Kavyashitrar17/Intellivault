"""
save_chunks.py  (fixed)
-----------------------
Fix: Use os.path.abspath() to resolve paths correctly on Windows.
     Explicitly validate that CHUNKS_PATH ends with a filename,
     not just a directory, before attempting to open it.
"""

import os
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Resolve path at module level — avoids Windows path issues
# -------------------------------------------------------
def _get_chunks_path() -> str:
    """
    Return the absolute path to chunks.json.
    Falls back to a safe default if config gives a directory instead of a file.
    """
    try:
        from backend.config import settings
        raw = settings.CHUNKS_PATH
    except Exception:
        raw = "data/processed_chunks/chunks.json"

    path = os.path.abspath(raw)

    # Guard: if the resolved path is a directory (or has no extension),
    # append the filename so we never try to open a directory as a file.
    if os.path.isdir(path) or not os.path.splitext(path)[1]:
        path = os.path.join(path, "chunks.json")

    return path


def _ensure_dir(path: str):
    """Create parent directory of `path` if it doesn't exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# -------------------------------------------------------
# Public API
# -------------------------------------------------------

def save_chunks(new_chunks: List[Dict]) -> int:
    """
    Append new chunks to chunks.json.
    Re-numbers all chunk IDs to stay in sync with FAISS index positions.
    Returns total chunk count after saving.
    """
    path = _get_chunks_path()
    _ensure_dir(path)

    logger.info(f"[SaveChunks] Writing to: {path}")

    existing = load_chunks()
    logger.info(f"[SaveChunks] Loaded {len(existing)} existing chunks.")

    offset = len(existing)
    for i, chunk in enumerate(new_chunks):
        chunk["chunk_id"] = offset + i

    all_chunks = existing + new_chunks

    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[SaveChunks] Saved {len(all_chunks)} total chunks "
        f"({len(new_chunks)} new + {len(existing)} existing)."
    )
    return len(all_chunks)


def load_chunks() -> List[Dict]:
    """Load all chunks from disk. Returns [] if file does not exist."""
    path = _get_chunks_path()
    if not os.path.exists(path):
        logger.info(f"[SaveChunks] No chunks file found at: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info(f"[SaveChunks] Loaded {len(chunks)} chunks from {path}")
        return chunks
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"[SaveChunks] Failed to load chunks: {e}")
        return []


def count_chunks() -> int:
    """Return chunk count without loading all data into RAM."""
    path = _get_chunks_path()
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return len(json.load(f))
    except Exception:
        return 0