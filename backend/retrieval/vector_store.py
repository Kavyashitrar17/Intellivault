import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings):
        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query_embedding, k=2):
        distances, indices = self.index.search(query_embedding, k)
        return indices
