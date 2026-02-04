from ingestion.loader import load_document
from ingestion.chunker import chunk_text

# Load long PDF
text = load_document("sample.txt")

print("Total characters:", len(text))

# Chunk the document
chunks = chunk_text(
    text=text,
    source="sample.txt",
    chunk_size=200,
    overlap=50
)

print("Total chunks created:", len(chunks))

# Print first 2 chunks for verification
for chunk in chunks[:3]:
    print("\n--- CHUNK", chunk["chunk_id"], "---")
    print(chunk["text"][:500])
