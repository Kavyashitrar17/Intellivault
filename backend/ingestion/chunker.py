"""
chunker.py
----------
Splits a document's text into smaller overlapping chunks.

WHY CHUNKING?
  LLMs and FAISS work best on short, focused pieces of text.
  We can't embed an entire 50-page PDF as one vector.

WHY OVERLAP?
  If an answer spans two adjacent chunks, overlap ensures it
  appears fully in at least one chunk.

CHANGES FROM ORIGINAL:
- Added MIN_WORDS filter: drops tiny/noisy chunks (e.g. headers, page numbers)
- Chunk IDs now include the source name: "notes_0", "notes_1" — easier to debug
- Added logging so you can see how many chunks each file produces
- Added a clean docstring explaining every parameter
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# --- Tunable constants ---
DEFAULT_CHUNK_SIZE    = 400   # words per chunk (not characters)
DEFAULT_OVERLAP       = 50    # words shared between adjacent chunks
MIN_WORDS_PER_CHUNK   = 20    # chunks shorter than this are likely noise


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict]:
    """
    Split text into overlapping word-based chunks with metadata.

    Args:
        text:       Raw document text (from loader.py).
        source:     Filename of the document (e.g. "lecture1.pdf").
        chunk_size: How many words per chunk.
        overlap:    How many words the next chunk re-uses from the previous one.
                    This prevents answers from being cut off at chunk boundaries.

    Returns:
        List of dicts. Each dict looks like:
            {
                "chunk_id": "lecture1_0",
                "source":   "lecture1.pdf",
                "text":     "Operating systems manage hardware..."
            }
    """
    if not text.strip():
        logger.warning(f"[Chunker] Empty text received for '{source}'. Skipping.")
        return []

    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0

    # A safe base name for chunk IDs (removes extension and spaces)
    base_name = source.replace(" ", "_").rsplit(".", 1)[0]

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        # Skip chunks that are too short — likely headers, page numbers, noise
        if len(chunk_words) >= MIN_WORDS_PER_CHUNK:
            chunks.append({
                "chunk_id": f"{base_name}_{chunk_index}",
                "source":   source,
                "text":     " ".join(chunk_words),
            })
            chunk_index += 1
        else:
            logger.debug(
                f"[Chunker] Skipped short chunk at word {start} "
                f"(only {len(chunk_words)} words)"
            )

        # Advance by (chunk_size - overlap) to create the sliding window
        step = chunk_size - overlap
        start += max(step, 1)  # prevent infinite loop if overlap >= chunk_size

    logger.info(
        f"[Chunker] '{source}' → {len(chunks)} chunks "
        f"(size={chunk_size} words, overlap={overlap} words)"
    )
    return chunks