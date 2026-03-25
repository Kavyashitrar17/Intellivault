"""
api.py
------
FastAPI backend for IntelliVault.

Endpoints:
  POST /upload   — Upload and index a document
  POST /query    — Ask a question against indexed documents
  GET  /status   — Check how many docs/chunks are indexed
  DELETE /reset  — Clear all indexed data (useful during development)

CHANGES FROM ORIGINAL:
  - Uses VectorStore class (retrieval/vector_store.py) instead of raw FAISS calls
  - /upload response now includes per-file feedback (chunk count, total vectors)
  - /query response now includes a "confidence" field for the UI
  - Added /reset endpoint for clearing data during testing
  - Moved all FAISS operations into VectorStore — api.py stays clean
  - Validates file type before processing
"""

import os
import shutil
import pickle
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.ingestion.loader      import load_document
from backend.ingestion.chunker     import chunk_text
from backend.ingestion.embedder    import create_embeddings
from backend.ingestion.save_chunks import save_chunks
from backend.retrieval.vector_store import VectorStore
from backend.rag_pipeline           import rag_pipeline

# -------------------------------------------------------
# Logging
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# App setup
# -------------------------------------------------------
app = FastAPI(
    title="IntelliVault API",
    description="RAG-based document question answering system",
    version="1.0.0",
)

# Allow Streamlit (running on port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
UPLOAD_FOLDER = "data/uploads"
CHUNKS_PATH   = "data/processed_chunks/chunks.pkl"
INDEX_PATH    = "data/vector_db/faiss_index.index"
ALLOWED_EXTS  = {".pdf", ".txt"}

os.makedirs(UPLOAD_FOLDER,                  exist_ok=True)
os.makedirs(os.path.dirname(CHUNKS_PATH),   exist_ok=True)
os.makedirs(os.path.dirname(INDEX_PATH),    exist_ok=True)


# -------------------------------------------------------
# Request schema
# -------------------------------------------------------
class QueryRequest(BaseModel):
    query: str


# -------------------------------------------------------
# POST /upload
# -------------------------------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and index a document.

    Steps:
    1. Validate file type (.pdf or .txt only)
    2. Save to disk
    3. Extract text (loader.py)
    4. Split into chunks (chunker.py)
    5. Create embeddings (embedder.py)
    6. Add to FAISS index (vector_store.py)
    7. Save chunks to disk (save_chunks.py)
    """
    # --- Validate extension ---
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only .pdf and .txt are allowed."
        )

    try:
        # 1. Save file to uploads folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"[Upload] Saved: {file_path}")

        # 2. Load text from file
        text = load_document(file_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="File is empty or unreadable.")

        # 3. Chunk the text
        chunks = chunk_text(text, source=file.filename)
        if not chunks:
            raise HTTPException(status_code=400, detail="No usable content found in file.")
        logger.info(f"[Upload] {len(chunks)} chunks created.")

        # 4. Embed the chunks
        embeddings = create_embeddings(chunks)

        # 5. Add to FAISS vector store
        store = VectorStore()
        store.add(embeddings)

        # 6. Save chunks to disk
        total_chunks = save_chunks(chunks)
        store.save()

        return {
            "message":       f"'{file.filename}' uploaded and indexed successfully.",
            "chunks_added":  len(chunks),
            "total_chunks":  total_chunks,
            "total_vectors": store.total_vectors,
        }

    except HTTPException:
        raise  # pass through our own HTTP errors

    except Exception as e:
        logger.error(f"[Upload] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# -------------------------------------------------------
# POST /query
# -------------------------------------------------------
@app.post("/query")
async def query_api(request: QueryRequest):
    """
    Ask a question. Returns answer, source previews, and confidence level.

    Response format:
        {
            "answer":       "...",
            "sources":      ["chunk preview...", ...],
            "source_files": ["notes.pdf", ...],
            "confidence":   "high" | "medium" | "low" | "none"
        }
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = rag_pipeline(request.query)
        return result

    except Exception as e:
        logger.error(f"[Query] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# -------------------------------------------------------
# GET /status
# -------------------------------------------------------
@app.get("/status")
def status():
    """
    Returns current system state.
    Useful for debugging and for the UI to show a health indicator.
    """
    store        = VectorStore()
    total_chunks = 0

    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            total_chunks = len(pickle.load(f))

    return {
        "status":        "ok",
        "total_vectors": store.total_vectors,
        "total_chunks":  total_chunks,
        "index_synced":  store.total_vectors == total_chunks,
    }


# -------------------------------------------------------
# DELETE /reset
# -------------------------------------------------------
@app.delete("/reset")
def reset():
    """
    Clear all indexed data (FAISS index + chunks file).
    Useful during development and testing.
    Does NOT delete uploaded files from data/uploads/.
    """
    deleted = []

    for path in [INDEX_PATH, CHUNKS_PATH]:
        if os.path.exists(path):
            os.remove(path)
            deleted.append(path)
            logger.info(f"[Reset] Deleted {path}")

    return {
        "message": "Index and chunks cleared.",
        "deleted": deleted,
    }