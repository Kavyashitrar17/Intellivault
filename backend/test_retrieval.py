import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("data/vector_db/faiss_index.index")

# Load chunks
with open("data/processed_chunks/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Query
query = "What is deadlock?"

query_embedding = model.encode([query]).astype("float32")
# Match how documents are indexed: normalized vectors.
norm = np.linalg.norm(query_embedding)
if norm > 0:
    query_embedding = query_embedding / norm

# Search
distances, indices = index.search(query_embedding, k=2)

print("\nRetrieved Chunks:\n")
for i in indices[0]:
    print("-", chunks[i]["text"], "\n")