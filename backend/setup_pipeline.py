from backend.ingestion.loader import load_document
from backend.ingestion.chunker import chunk_text
from backend.ingestion.embedder import create_embeddings
from backend.retrieval.vector_store import VectorStore

# 1. Load document
doc = load_document("sample.txt")

# 2. Chunk document
chunks = chunk_text(doc, "sample.txt")

print(f"Total chunks created: {len(chunks)}")

import pickle
import os



# Save chunks
with open("data/processed_chunks/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Chunks saved successfully!")
# 3. Generate embeddings
embeddings = create_embeddings(chunks)

# 4. Create vector store
vector_store = VectorStore()
vector_store.add(embeddings)
vector_store.save()

print("✅ Vector DB created successfully!")