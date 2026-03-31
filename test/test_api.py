"""
tests/test_api.py
-----------------
Integration tests for all FastAPI endpoints.

Uses FastAPI's TestClient (synchronous) so no async setup needed.
The heavy stuff (embedder, LLM, FAISS) is mocked for speed.
"""

import io
import os
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.retrieval.vector_store import reset_vector_store


@pytest.fixture(autouse=True)
def reset_store():
    reset_vector_store()
    yield
    reset_vector_store()


@pytest.fixture
def client(tmp_data_dir):
    """
    Fresh TestClient with all external calls mocked:
    - Embedding model returns deterministic fake vectors
    - rag_pipeline returns a fixed dict
    """
    fake_emb = np.random.rand(1, 384).astype("float32")
    fake_emb /= np.linalg.norm(fake_emb, axis=1, keepdims=True)

    with patch("backend.ingestion.embedder.get_model") as mock_model, \
         patch("backend.rag_pipeline.rag_pipeline") as mock_pipeline:

        # Embedding model mock
        model_instance = MagicMock()
        model_instance.encode.return_value = fake_emb
        mock_model.return_value = model_instance

        # Pipeline mock
        mock_pipeline.return_value = {
            "answer":       "A deadlock occurs when processes wait indefinitely.",
            "sources":      ["A deadlock is a situation..."],
            "source_files": ["os_notes.pdf"],
            "confidence":   "high",
        }

        from backend.api import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, mock_pipeline


# -------------------------------------------------------
# POST /upload
# -------------------------------------------------------
class TestUpload:

    def test_upload_txt_succeeds(self, client, tmp_data_dir):
        c, _ = client
        txt_content = b"This is a test document about operating systems and deadlocks."
        response = c.post(
            "/upload",
            files={"file": ("test.txt", io.BytesIO(txt_content), "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "chunks_added"  in data
        assert "total_vectors" in data
        assert "total_chunks"  in data

    def test_upload_rejects_unsupported_extension(self, client):
        c, _ = client
        response = c.post(
            "/upload",
            files={"file": ("malware.exe", io.BytesIO(b"bad"), "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_rejects_empty_file(self, client):
        c, _ = client
        response = c.post(
            "/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_rejects_oversized_file(self, client):
        c, _ = client
        big = b"x" * (21 * 1024 * 1024)  # 21 MB
        response = c.post(
            "/upload",
            files={"file": ("big.txt", io.BytesIO(big), "text/plain")},
        )
        assert response.status_code == 413

    def test_upload_response_has_filename_in_message(self, client):
        c, _ = client
        content = b"Some meaningful content about deadlocks and scheduling algorithms."
        response = c.post(
            "/upload",
            files={"file": ("notes.txt", io.BytesIO(content), "text/plain")},
        )
        if response.status_code == 200:
            assert "notes.txt" in response.json()["message"]


# -------------------------------------------------------
# POST /query
# -------------------------------------------------------
class TestQuery:

    def test_query_returns_200(self, client, sample_chunks):
        c, mock_pipeline = client
        response = c.post("/query", json={"query": "What is a deadlock?"})
        assert response.status_code == 200

    def test_query_response_has_required_keys(self, client):
        c, _ = client
        response = c.post("/query", json={"query": "What is a deadlock?"})
        data = response.json()
        assert "answer"       in data
        assert "sources"      in data
        assert "source_files" in data
        assert "confidence"   in data

    def test_query_empty_string_returns_400(self, client):
        c, _ = client
        response = c.post("/query", json={"query": ""})
        assert response.status_code == 400

    def test_query_whitespace_only_returns_400(self, client):
        c, _ = client
        response = c.post("/query", json={"query": "   "})
        assert response.status_code == 400

    def test_query_missing_field_returns_422(self, client):
        c, _ = client
        response = c.post("/query", json={})
        assert response.status_code == 422

    def test_query_answer_is_string(self, client):
        c, _ = client
        response = c.post("/query", json={"query": "What is a deadlock?"})
        assert isinstance(response.json()["answer"], str)

    def test_query_confidence_valid_value(self, client):
        c, _ = client
        response = c.post("/query", json={"query": "What is a deadlock?"})
        assert response.json()["confidence"] in {"high", "medium", "low", "none"}


# -------------------------------------------------------
# GET /status
# -------------------------------------------------------
class TestStatus:

    def test_status_returns_200(self, client):
        c, _ = client
        response = c.get("/status")
        assert response.status_code == 200

    def test_status_has_required_fields(self, client):
        c, _ = client
        data = c.get("/status").json()
        assert "status"        in data
        assert "total_vectors" in data
        assert "total_chunks"  in data
        assert "index_synced"  in data

    def test_status_value_is_ok(self, client):
        c, _ = client
        assert c.get("/status").json()["status"] == "ok"

    def test_index_synced_is_bool(self, client):
        c, _ = client
        assert isinstance(c.get("/status").json()["index_synced"], bool)


# -------------------------------------------------------
# DELETE /reset
# -------------------------------------------------------
class TestReset:

    def test_reset_returns_200(self, client):
        c, _ = client
        response = c.delete("/reset")
        assert response.status_code == 200

    def test_reset_response_has_message(self, client):
        c, _ = client
        data = c.delete("/reset").json()
        assert "message" in data
        assert "deleted"  in data

    def test_reset_clears_chunks_file(self, client):
        from backend.config import settings
        c, _ = client
        # Write a fake chunks file
        os.makedirs(os.path.dirname(settings.CHUNKS_PATH), exist_ok=True)
        with open(settings.CHUNKS_PATH, "w") as f:
            json.dump([], f)
        c.delete("/reset")
        assert not os.path.exists(settings.CHUNKS_PATH)


# -------------------------------------------------------
# API key auth
# -------------------------------------------------------
class TestApiKeyAuth:

    def test_no_auth_required_when_key_not_set(self, client):
        """With API_KEY="" in settings, all requests go through."""
        c, _ = client
        response = c.post("/query", json={"query": "test"})
        assert response.status_code != 403

    def test_wrong_key_returns_403(self, tmp_data_dir):
        from backend import config
        with patch.object(config.settings, "API_KEY", "secret123"):
            from backend.api import app
            with TestClient(app) as c:
                response = c.post(
                    "/query",
                    json={"query": "test"},
                    headers={"X-API-Key": "wrongkey"},
                )
                assert response.status_code == 403

    def test_correct_key_passes(self, tmp_data_dir):
        from backend import config
        with patch.object(config.settings, "API_KEY", "secret123"), \
             patch("backend.rag_pipeline.rag_pipeline", return_value={
                 "answer": "ok", "sources": [], "source_files": [], "confidence": "low"
             }):
            from backend.api import app
            with TestClient(app) as c:
                response = c.post(
                    "/query",
                    json={"query": "test"},
                    headers={"X-API-Key": "secret123"},
                )
                assert response.status_code == 200
