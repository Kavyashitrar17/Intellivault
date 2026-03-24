import os
import shutil
import logging
import numpy as np
import faiss
import pickle

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.ingestion.loader      import load_document
from backend.ingestion.chunker     import chunk_text
from backend.ingestion.embedder    import create_embeddings
from backend.ingestion.save_chunks import save_chunks
from backend.rag_pipeline          import rag_pipeline

# -------------------------------------------------------
# Logging
# -------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------
# FastAPI app
# -------------------------------------------------------
app = FastAPI(title="IntelliVault API")

# -------------------------------------------------------
# Paths — same as rag_pipeline.py so they always match
# -------------------------------------------------------
UPLOAD_FOLDER = "data/uploads"
INDEX_PATH    = "data/vector_db/faiss_index.index"
CHUNKS_PATH   = "data/processed_chunks/chunks.pkl"
DIMENSION     = 384   # must match all-MiniLM-L6-v2 output size

# Make sure folders exist at startup
os.makedirs(UPLOAD_FOLDER,                     exist_ok=True)
os.makedirs(os.path.dirname(INDEX_PATH),       exist_ok=True)
os.makedirs(os.path.dirname(CHUNKS_PATH),      exist_ok=True)


# -------------------------------------------------------
# Request body model for /query
# -------------------------------------------------------
class QueryRequest(BaseModel):
    query: str


# -------------------------------------------------------
# POST /upload
# -------------------------------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file.
    Pipeline: save → load text → chunk → embed → store in FAISS + pkl
    """
    try:
        # 1. Save uploaded file to disk
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to {file_path}")

        # 2. Load text from PDF or TXT
        text = load_document(file_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty or unreadable.")

        # 3. Split text into overlapping chunks
        chunks = chunk_text(text, source=file.filename)
        logger.info(f"Created {len(chunks)} chunks from {file.filename}")

        # 4. Create embeddings for all chunks
        embeddings = create_embeddings(chunks)
        embeddings_np = np.array(embeddings).astype("float32")

        # 5. Load existing FAISS index OR create a new one (cold start safe)
        if os.path.exists(INDEX_PATH):
            index = faiss.read_index(INDEX_PATH)
            logger.info(f"Loaded existing FAISS index with {index.ntotal} vectors.")
        else:
            logger.warning("No existing index — creating new FAISS index.")
            index = faiss.IndexFlatL2(DIMENSION)

        # 6. Add new embeddings to the index
        index.add(embeddings_np)
        logger.info(f"Added {len(embeddings_np)} vectors. Index now has {index.ntotal} total.")

        # 7. Save updated FAISS index back to disk
        faiss.write_index(index, INDEX_PATH)
        logger.info(f"Saved FAISS index to {INDEX_PATH}")

        # 8. Save chunks to pkl (save_chunks handles appending)
        save_chunks(chunks)

        return {
            "message":      f"{file.filename} uploaded and indexed successfully.",
            "chunks_added": len(chunks),
            "total_vectors": index.ntotal
        }

    except HTTPException:
        raise   # re-raise our own HTTP errors as-is

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# -------------------------------------------------------
# POST /query
# -------------------------------------------------------
@app.post("/query")
async def query_api(request: QueryRequest):
    """
    Ask a question against the uploaded documents.
    Returns: { answer: str, sources: list[str] }
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        result = rag_pipeline(request.query)
        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# -------------------------------------------------------
# GET /status  — quick health check (useful for debugging)
# -------------------------------------------------------
@app.get("/status")
def status():
    """
    Returns how many vectors and chunks are currently stored.
    Useful to confirm uploads worked without querying.
    """
    index_exists  = os.path.exists(INDEX_PATH)
    chunks_exist  = os.path.exists(CHUNKS_PATH)

    total_vectors = 0
    total_chunks  = 0

    if index_exists:
        index         = faiss.read_index(INDEX_PATH)
        total_vectors = index.ntotal

    if chunks_exist:
        with open(CHUNKS_PATH, "rb") as f:
            total_chunks = len(pickle.load(f))

    return {
        "index_exists":  index_exists,
        "chunks_exist":  chunks_exist,
        "total_vectors": total_vectors,
        "total_chunks":  total_chunks,
    }