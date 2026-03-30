"""
chunker.py
----------
IMPROVEMENTS OVER ORIGINAL:
  1. Sentence-aware chunking — chunks now always end at a sentence boundary.
     Original split on raw word count, so answers could be cut mid-sentence.
     Now we split into sentences first, then group them into word-count windows.
  2. Falls back gracefully if nltk is not installed (uses regex splitting).
  3. All constants come from config.py.
  4. Same metadata format as before — drop-in replacement.

Install for best results:  pip install nltk
Then run once:             python -m nltk.downloader punkt
"""

import logging
import re
from typing import List, Dict

from backend.config import settings

logger = logging.getLogger(__name__)

# Try to use nltk for proper sentence tokenization
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
    _USE_NLTK = True
except (ImportError, LookupError):
    _USE_NLTK = False
    logger.info(
        "[Chunker] nltk punkt not available — using regex sentence splitter. "
        "For better chunking: pip install nltk && python -m nltk.downloader punkt"
    )


def _split_sentences(text: str) -> List[str]:
    """Split text into individual sentences."""
    if _USE_NLTK:
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    # Regex fallback: split on ". ", "? ", "! ", or newlines
    raw = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in raw if s.strip()]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = None,
    overlap: int = None,
) -> List[Dict]:
    """
    Split text into overlapping, sentence-boundary-respecting chunks.

    Args:
        text:       Raw document text.
        source:     Filename (e.g. "lecture1.pdf").
        chunk_size: Target words per chunk (default from config).
        overlap:    Overlap words between adjacent chunks (default from config).

    Returns:
        List of chunk dicts: {chunk_id, source, text}
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap    = overlap    or settings.CHUNK_OVERLAP

    if not text.strip():
        logger.warning(f"[Chunker] Empty text for '{source}'.")
        return []

    sentences     = _split_sentences(text)
    base_name     = source.replace(" ", "_").rsplit(".", 1)[0]
    chunks        = []
    chunk_index   = 0
    current_words = []   # words in the current chunk
    current_sents = []   # sentences in the current chunk (for overlap)

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        # If adding this sentence would overflow the chunk, flush first
        if current_words and len(current_words) + len(words) > chunk_size:
            chunk_text_str = " ".join(current_words)
            if len(current_words) >= settings.MIN_CHUNK_WORDS:
                chunks.append({
                    "chunk_id": f"{base_name}_{chunk_index}",
                    "source":   source,
                    "text":     chunk_text_str,
                })
                chunk_index += 1

            # Carry over the last `overlap` words as context for the next chunk
            overlap_text = " ".join(current_words[-overlap:]) if overlap else ""
            current_words = overlap_text.split() if overlap_text else []
            current_sents = []

        current_words.extend(words)
        current_sents.append(sentence)

    # Flush the last chunk
    if len(current_words) >= settings.MIN_CHUNK_WORDS:
        chunks.append({
            "chunk_id": f"{base_name}_{chunk_index}",
            "source":   source,
            "text":     " ".join(current_words),
        })

    logger.info(
        f"[Chunker] '{source}' → {len(chunks)} chunks "
        f"(size≈{chunk_size} words, overlap={overlap} words, "
        f"sentence-aware={'yes' if _USE_NLTK else 'regex'})"
    )
    return chunks