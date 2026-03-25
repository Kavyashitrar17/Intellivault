"""
loader.py
---------
Loads raw text from uploaded PDF or TXT files.

CHANGES FROM ORIGINAL:
- Switched from pypdf → PyMuPDF (fitz): better real-world PDF text extraction
- Added logging so you see what's happening in the terminal
- Skips blank/image-only PDF pages instead of crashing
- More specific error messages
"""

import os
import logging

logger = logging.getLogger(__name__)


def load_document(file_path: str) -> str:
    """
    Load and return raw text from a .txt or .pdf file.

    Args:
        file_path: Full path to the file on disk.

    Returns:
        Extracted text as one string.

    Raises:
        FileNotFoundError: File doesn't exist.
        ValueError: File type is not .pdf or .txt.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return _load_txt(file_path)
    elif ext == ".pdf":
        return _load_pdf(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Only .pdf and .txt are allowed."
        )


def _load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    logger.info(f"[Loader] TXT loaded: {os.path.basename(file_path)} ({len(text)} chars)")
    return text.strip()


def _load_pdf(file_path: str) -> str:
    """
    Extract text using PyMuPDF (fitz).
    Falls back page-by-page and skips blank pages.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    doc = fitz.open(file_path)
    pages_text = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages_text.append(text)
        else:
            logger.debug(f"[Loader] Skipping blank page {i+1}")

    num_pages = len(doc)
    doc.close()
    full_text = "\n\n".join(pages_text)
    logger.info(
        f"[Loader] PDF loaded: {os.path.basename(file_path)} | "
        f"{num_pages} pages | {len(full_text)} chars"
    )
    return full_text.strip()