"""
conftest.py
-----------
Shared pytest fixtures used across all test files.

Run all tests:       pytest tests/ -v
Run with coverage:   pytest tests/ -v --cov=backend --cov-report=term-missing
"""

import os
import json
import shutil
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# -------------------------------------------------------
# Temp directory — isolated per test session
# -------------------------------------------------------
@pytest.fixture(scope="session")
def tmp_data_dir():
    """Create a fresh temp directory for all data files, clean up after."""
    d = tempfile.mkdtemp(prefix="intellivault_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def patch_settings(tmp_data_dir, monkeypatch):
    """
    Redirect all file paths to the temp directory so tests never
    touch real project data. Applied automatically to every test.
    """
    upload_dir  = os.path.join(tmp_data_dir, "uploads")
    chunks_path = os.path.join(tmp_data_dir, "chunks.json")
    index_path  = os.path.join(tmp_data_dir, "faiss_index.index")

    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(os.path.dirname(chunks_path), exist_ok=True)

    from backend import config
    monkeypatch.setattr(config.settings, "UPLOAD_FOLDER", upload_dir)
    monkeypatch.setattr(config.settings, "CHUNKS_PATH",   chunks_path)
    monkeypatch.setattr(config.settings, "INDEX_PATH",    index_path)
    monkeypatch.setattr(config.settings, "API_KEY",       "")  # auth off


# -------------------------------------------------------
# Sample data
# -------------------------------------------------------
@pytest.fixture
def sample_text():
    return (
        "A deadlock is a situation where two or more processes are unable to proceed "
        "because each is waiting for the other to release a resource. "
        "Deadlocks can be prevented by using resource ordering or timeouts. "
        "Operating systems often use banker's algorithm to avoid deadlocks. "
        "Another approach is deadlock detection and recovery. "
        "Semaphores and mutexes are common synchronization primitives used in OS. "
        "Process scheduling determines the order in which processes run on the CPU. "
        "Virtual memory allows processes to use more memory than physically available."
    )


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "doc_0",
            "source":   "os_notes.pdf",
            "text":     (
                "A deadlock is a situation where two or more processes are unable to "
                "proceed because each is waiting for the other to release a resource."
            ),
        },
        {
            "chunk_id": "doc_1",
            "source":   "os_notes.pdf",
            "text":     (
                "Deadlocks can be prevented by using resource ordering or timeouts. "
                "The banker's algorithm is used to avoid unsafe states."
            ),
        },
        {
            "chunk_id": "doc_2",
            "source":   "os_notes.pdf",
            "text":     (
                "Process scheduling determines the order in which processes run on "
                "the CPU. Round-robin and priority scheduling are common algorithms."
            ),
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Reproducible fake embeddings (normalized)."""
    np.random.seed(42)
    emb = np.random.rand(3, 384).astype("float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


# -------------------------------------------------------
# FAISS store with pre-loaded embeddings
# -------------------------------------------------------
@pytest.fixture
def loaded_store(sample_embeddings, tmp_data_dir):
    """A VectorStore with 3 vectors already added."""
    from backend.retrieval.vector_store import VectorStore, reset_vector_store
    reset_vector_store()

    index_path = os.path.join(tmp_data_dir, "faiss_index.index")
    store = VectorStore(index_path=index_path)
    store.add(sample_embeddings)
    store.save()
    return store


# -------------------------------------------------------
# FastAPI test client
# -------------------------------------------------------
@pytest.fixture
def api_client(loaded_store):
    """
    TestClient for the FastAPI app.
    Embedding model and vector store are mocked so tests run fast.
    """
    from backend.retrieval import vector_store as vs_module

    # Patch get_vector_store to return our pre-loaded store
    with patch.object(vs_module, "get_vector_store", return_value=loaded_store):
        from backend.api import app
        with TestClient(app) as client:
            yield client
