from ingestion.loader import load_document
from ingestion.chunker import chunk_text
from ingestion.embedder import create_embeddings   
from retrieval.vector_store import VectorStore

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
vector_store = VectorStore(384)
vector_store.add_embeddings(embeddings)

# 5. SAVE FAISS (VERY IMPORTANT)
import faiss
faiss.write_index(vector_store.index, "data/vector_db/faiss_index.index")

print("✅ Vector DB created successfully!")