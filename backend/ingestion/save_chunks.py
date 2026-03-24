import os
import pickle
import logging

# Set up logging so we can see what's happening in the terminal
logger = logging.getLogger(__name__)

# ✅ Correct path — matches what rag_pipeline.py expects
CHUNKS_PATH = "data/processed_chunks/chunks.pkl"


def save_chunks(new_chunks: list):
    """
    Appends new chunks to the existing chunks.pkl file.
    If the file doesn't exist yet, it creates it (cold start safe).

    Args:
        new_chunks: list of chunk dicts with keys: chunk_id, source, text
    """

    # Make sure the folder exists
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

    # Load existing chunks if file already exists
    existing_chunks = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            existing_chunks = pickle.load(f)
        logger.info(f"Loaded {len(existing_chunks)} existing chunks from disk.")

    # Re-number chunk IDs so they stay unique across uploads
    offset = len(existing_chunks)
    for i, chunk in enumerate(new_chunks):
        chunk["chunk_id"] = offset + i

    # Combine old + new and save
    all_chunks = existing_chunks + new_chunks

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    logger.info(f"Saved {len(all_chunks)} total chunks to {CHUNKS_PATH}")
    return all_chunks