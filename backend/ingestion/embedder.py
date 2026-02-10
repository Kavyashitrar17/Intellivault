from sentence_transformers import SentenceTransformer

# Load small but powerful model
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):
    """
    Converts text chunks into vector embeddings.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    return embeddings
