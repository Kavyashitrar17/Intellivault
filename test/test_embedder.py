"""
tests/test_embedder.py
----------------------
Tests for backend/ingestion/embedder.py
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestGetModel:

    def test_get_model_returns_same_instance(self):
        from backend.ingestion.embedder import get_model
        m1 = get_model()
        m2 = get_model()
        assert m1 is m2, "get_model() must return the same singleton instance"

    def test_model_is_not_none(self):
        from backend.ingestion.embedder import get_model
        assert get_model() is not None


class TestCreateEmbeddings:

    def test_output_shape(self, sample_chunks):
        from backend.ingestion.embedder import create_embeddings
        emb = create_embeddings(sample_chunks)
        assert emb.shape == (len(sample_chunks), 384)

    def test_output_dtype_float32(self, sample_chunks):
        from backend.ingestion.embedder import create_embeddings
        emb = create_embeddings(sample_chunks)
        assert emb.dtype == np.float32

    def test_embeddings_are_normalized(self, sample_chunks):
        """Each row should have L2 norm ≈ 1.0."""
        from backend.ingestion.embedder import create_embeddings
        emb = create_embeddings(sample_chunks)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_empty_input_returns_empty_array(self):
        from backend.ingestion.embedder import create_embeddings
        result = create_embeddings([])
        assert len(result) == 0

    def test_different_texts_produce_different_embeddings(self):
        from backend.ingestion.embedder import create_embeddings
        chunks = [
            {"text": "deadlock in operating systems"},
            {"text": "machine learning neural networks"},
        ]
        emb = create_embeddings(chunks)
        # Cosine similarity should be well below 1 for unrelated texts
        sim = float(np.dot(emb[0], emb[1]))
        assert sim < 0.95, "Unrelated texts should have different embeddings"

    def test_similar_texts_have_high_similarity(self):
        from backend.ingestion.embedder import create_embeddings
        chunks = [
            {"text": "What is a deadlock in operating systems?"},
            {"text": "A deadlock occurs when processes cannot proceed."},
        ]
        emb = create_embeddings(chunks)
        sim = float(np.dot(emb[0], emb[1]))
        assert sim > 0.5, "Similar texts should have high cosine similarity"


class TestEmbedQuery:

    def test_output_shape(self):
        from backend.ingestion.embedder import embed_query
        emb = embed_query("What is a deadlock?")
        assert emb.shape == (1, 384)

    def test_output_dtype_float32(self):
        from backend.ingestion.embedder import embed_query
        emb = embed_query("test query")
        assert emb.dtype == np.float32

    def test_query_is_normalized(self):
        from backend.ingestion.embedder import embed_query
        emb = embed_query("What is process scheduling?")
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, "Query embedding should be L2-normalized"
