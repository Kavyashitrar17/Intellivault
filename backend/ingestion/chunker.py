def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 400,
    overlap: int = 50
):
    """
    Splits text into overlapping chunks with source metadata.
    """

    words = text.split()
    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "source": source,
            "text": " ".join(chunk_words)
        })

        chunk_id += 1
        start += chunk_size - overlap

    return chunks
