from backend.ingestion.loader import load_document
from backend.ingestion.chunker import chunk_text
from backend.ingestion.embedder import create_embeddings
from backend.retrieval.vector_store import VectorStore
from sentence_transformers import SentenceTransformer

# Load document
text = load_document("sample.txt")
print("Total characters:", len(text))

# Chunking
chunks = chunk_text(
    text=text,
    source="sample.txt",
    chunk_size=200,
    overlap=50
)

print("Total chunks created:", len(chunks))

# Preview chunks
for chunk in chunks[:3]:
    print("\n--- CHUNK", chunk["chunk_id"], "---")
    print(chunk["text"][:200])

# Create embeddings
embeddings = create_embeddings(chunks)
print("Embedding shape:", embeddings.shape)

# Vector store
store = VectorStore()
store.add(embeddings)
store.save()

# Query
model = SentenceTransformer("all-MiniLM-L6-v2")
query = "What is deadlock?"
query_embedding = model.encode([query]).astype("float32")
# Index uses inner product over normalized embeddings; normalize query too.
import numpy as np
norm = np.linalg.norm(query_embedding)
if norm > 0:
    query_embedding = query_embedding / norm

results = store.search(query_embedding)

print("\nMost relevant chunk index:", results)
print("\nRelevant text:\n", chunks[results[0][0]]["text"])