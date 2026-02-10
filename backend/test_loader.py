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

from ingestion.embedder import create_embeddings

embeddings = create_embeddings(chunks)
print("Embedding shape:", embeddings.shape)


from retrieval.vector_store import VectorStore
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# create vector DB
store = VectorStore(embeddings.shape[1])
store.add_embeddings(embeddings)

# ask a question
query = "What is deadlock?"
query_embedding = model.encode([query])

results = store.search(query_embedding)

print("Most relevant chunk index:", results)
print("Relevant text:\n", chunks[results[0][0]]["text"])


