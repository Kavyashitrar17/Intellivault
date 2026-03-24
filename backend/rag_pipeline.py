import os
import faiss
import pickle
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------
# Logging
# -------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
INDEX_PATH  = "data/vector_db/faiss_index.index"
CHUNKS_PATH = "data/processed_chunks/chunks.pkl"
DIMENSION   = 384   # all-MiniLM-L6-v2 always outputs 384-dim vectors

# -------------------------------------------------------
# Load embedding model ONCE — read-only, safe at module level
# -------------------------------------------------------
logger.info("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Model loaded.")


# -------------------------------------------------------
# Load FAISS index from disk (cold start safe)
# -------------------------------------------------------
def load_index():
    if os.path.exists(INDEX_PATH):
        logger.info(f"Loading FAISS index from {INDEX_PATH}")
        return faiss.read_index(INDEX_PATH)
    else:
        logger.warning("No FAISS index found — creating empty index.")
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        return faiss.IndexFlatL2(DIMENSION)


# -------------------------------------------------------
# Load chunks list from disk (cold start safe)
# -------------------------------------------------------
def load_chunks():
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from disk.")
        return chunks
    else:
        logger.warning("No chunks file found — returning empty list.")
        return []


# -------------------------------------------------------
# Retrieve top-k relevant chunks for a query
# Returns: (list of chunk texts, list of source filenames)
# -------------------------------------------------------
def retrieve(query, k=5):
    """
    Embeds the query and finds the top-k nearest chunks in FAISS.
    Also returns the source filename for each chunk so the user
    can see which document the answer came from.
    """
    index  = load_index()
    chunks = load_chunks()

    # Cold start — nothing uploaded yet
    if index.ntotal == 0 or len(chunks) == 0:
        logger.warning("Index or chunks are empty — nothing to retrieve.")
        return [], []

    # Never ask for more results than exist in the index
    k = min(k, index.ntotal)

    # Embed the query — FAISS requires float32
    query_embedding = model.encode([query]).astype("float32")

    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)

    logger.info(f"FAISS returned indices: {indices[0]}")
    logger.info(f"Total chunks available: {len(chunks)}")

    texts   = []
    sources = []

    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            texts.append(chunks[idx]["text"])
            sources.append(chunks[idx].get("source", "unknown"))
        else:
            # Index mismatch guard — skip invalid positions
            logger.warning(f"Index {idx} out of range — skipping.")

    return texts, sources


# -------------------------------------------------------
# Generate an extractive answer from retrieved chunks
# -------------------------------------------------------
def generate_answer(query, retrieved_chunks):
    """
    Extractive answer generation — no LLM needed.

    How it works:
      1. Split each chunk into sentences.
      2. Score each sentence by how many query words it contains.
      3. Return the highest-scoring sentence.

    Why not GPT-2?
      GPT-2 repeats itself and hallucinates.
      This approach is fast, accurate, and grounded in the document.
    """
    if not retrieved_chunks:
        return "No relevant content found. Please upload a document first."

    # Use a set for fast word lookup
    query_words = set(query.lower().split())

    best_sentence = ""
    max_score     = -1

    for chunk in retrieved_chunks:
        # Split on period or newline to get individual sentences
        sentences = [s.strip() for s in chunk.replace("\n", ". ").split(".")]

        for sentence in sentences:
            # Skip tiny meaningless fragments
            if len(sentence) < 15:
                continue

            sentence_words = set(sentence.lower().split())

            # Score = number of query words that appear in this sentence
            score = len(query_words & sentence_words)

            if score > max_score:
                max_score     = score
                best_sentence = sentence

    # Fallback: return start of first chunk if nothing matched
    if not best_sentence:
        best_sentence = retrieved_chunks[0][:300]

    return best_sentence.strip()


# -------------------------------------------------------
# Main RAG pipeline — called by api.py /query endpoint
# -------------------------------------------------------
def rag_pipeline(query):
    """
    Full pipeline:
      1. Retrieve relevant chunks from FAISS
      2. Generate an extractive answer
      3. Return clean JSON with answer + source info

    Response format:
      {
        "answer":       "...",
        "sources":      ["first 150 chars...", ...],
        "source_files": ["lecture1.pdf", ...]
      }
    """
    logger.info(f"Running RAG pipeline for query: '{query}'")

    retrieved_chunks, source_files = retrieve(query)
    logger.info(f"Retrieved {len(retrieved_chunks)} chunk(s).")

    answer = generate_answer(query, retrieved_chunks)
    logger.info(f"Answer: {answer[:120]}")

    # Trim sources to first 150 chars so the response stays readable
    source_previews = [
        chunk[:150] + "..." if len(chunk) > 150 else chunk
        for chunk in retrieved_chunks
    ]

    return {
        "answer":       answer,
        "sources":      source_previews,        # short preview of matching chunks
        "source_files": list(set(source_files)) # which documents were used
    }