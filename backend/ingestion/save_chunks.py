"""
save_chunks.py
--------------
Saves (and appends) processed chunks to a .pkl file on disk.

HOW IT WORKS:
  Each time a new document is uploaded, its chunks are ADDED to the
  existing chunks file. This way, multiple documents build up over time.

IMPORTANT: chunk_id is re-numbered globally so IDs stay unique
across multiple uploads. chunk_id must match the FAISS vector index.

CHANGES FROM ORIGINAL:
- No logic changes — this was already correct
- Added clearer logging messages
- Added return value (total chunk count) so api.py can use it
"""

import os
import pickle
import logging

logger = logging.getLogger(__name__)

CHUNKS_PATH = "data/processed_chunks/chunks.pkl"


def save_chunks(new_chunks: list) -> int:
    """
    Append new chunks to chunks.pkl.
    Re-numbers all chunk IDs to keep them in sync with FAISS index positions.

    Args:
        new_chunks: List of chunk dicts (from chunker.py).

    Returns:
        Total number of chunks now stored (old + new).
    """
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

    # Load existing chunks if file exists
    existing_chunks = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            existing_chunks = pickle.load(f)
        logger.info(f"[SaveChunks] Loaded {len(existing_chunks)} existing chunks.")

    # Re-number new chunk IDs starting from where the old ones left off.
    # This is CRITICAL — chunk position in the list must equal its FAISS vector index.
    offset = len(existing_chunks)
    for i, chunk in enumerate(new_chunks):
        chunk["chunk_id"] = offset + i

    all_chunks = existing_chunks + new_chunks

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    logger.info(
        f"[SaveChunks] Saved {len(all_chunks)} total chunks "
        f"({len(new_chunks)} new + {len(existing_chunks)} existing)."
    )
    return len(all_chunks)