from pypdf import PdfReader


import os

def load_document(file_path: str) -> str:
    """
    Loads text from a PDF or TXT file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

    if file_path.endswith(".pdf"):
        return _load_pdf(file_path)

    elif file_path.endswith(".txt"):
        return _load_txt(file_path)

    else:
        raise ValueError("Unsupported file format")


def _load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def _load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()
