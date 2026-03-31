"""
tests/test_rag_pipeline.py
--------------------------
Tests for backend/rag_pipeline.py

The embedding model and LLM are mocked so these tests run fast
without downloading any models.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from backend.retrieval.vector_store import reset_vector_store


@pytest.fixture(autouse=True)
def reset_store():
    reset_vector_store()
    yield
    reset_vector_store()


@pytest.fixture
def mock_retrieve_results(sample_chunks):
    """Fake top_chunks with scores — what retriever.retrieve() returns."""
    return [
        {**sample_chunks[0], "score": 0.75, "semantic_score": 0.80, "keyword_score": 0.60},
        {**sample_chunks[1], "score": 0.55, "semantic_score": 0.60, "keyword_score": 0.45},
    ]


class TestRagPipeline:

    def test_returns_dict_with_required_keys(self, sample_chunks, mock_retrieve_results):
        from backend.rag_pipeline import rag_pipeline
        with patch("backend.rag_pipeline.load_chunks",       return_value=sample_chunks), \
             patch("backend.rag_pipeline._sync_if_needed",   return_value=None), \
             patch("backend.rag_pipeline.retrieve",          return_value=mock_retrieve_results), \
             patch("backend.rag_pipeline.generate_answer",   return_value="A deadlock occurs when processes wait."):
            result = rag_pipeline("What is a deadlock?")

        assert "answer"       in result
        assert "sources"      in result
        assert "source_files" in result
        assert "confidence"   in result

    def test_no_chunks_returns_none_confidence(self):
        from backend.rag_pipeline import rag_pipeline
        with patch("backend.rag_pipeline.load_chunks", return_value=[]):
            result = rag_pipeline("What is a deadlock?")
        assert result["confidence"]   == "none"
        assert result["sources"]      == []
        assert result["source_files"] == []

    def test_no_retrieval_results_returns_low_confidence(self, sample_chunks):
        from backend.rag_pipeline import rag_pipeline
        with patch("backend.rag_pipeline.load_chunks",     return_value=sample_chunks), \
             patch("backend.rag_pipeline._sync_if_needed", return_value=None), \
             patch("backend.rag_pipeline.retrieve",        return_value=[]):
            result = rag_pipeline("xyz unknown query")
        assert result["confidence"] == "low"

    def test_high_confidence_when_top_score_above_threshold(self, sample_chunks, mock_retrieve_results):
        from backend.rag_pipeline import rag_pipeline
        # top score = 0.75 → should be "high"
        with patch("backend.rag_pipeline.load_chunks",       return_value=sample_chunks), \
             patch("backend.rag_pipeline._sync_if_needed",   return_value=None), \
             patch("backend.rag_pipeline.retrieve",          return_value=mock_retrieve_results), \
             patch("backend.rag_pipeline.generate_answer",   return_value="Test answer."):
            result = rag_pipeline("deadlock?")
        assert result["confidence"] == "high"

    def test_medium_confidence_for_mid_score(self, sample_chunks):
        from backend.rag_pipeline import rag_pipeline
        mid_chunks = [
            {**sample_chunks[0], "score": 0.45, "semantic_score": 0.5, "keyword_score": 0.3},
        ]
        with patch("backend.rag_pipeline.load_chunks",       return_value=sample_chunks), \
             patch("backend.rag_pipeline._sync_if_needed",   return_value=None), \
             patch("backend.rag_pipeline.retrieve",          return_value=mid_chunks), \
             patch("backend.rag_pipeline.generate_answer",   return_value="Some answer."):
            result = rag_pipeline("deadlock?")
        assert result["confidence"] == "medium"

    def test_source_files_extracted_correctly(self, sample_chunks, mock_retrieve_results):
        from backend.rag_pipeline import rag_pipeline
        with patch("backend.rag_pipeline.load_chunks",       return_value=sample_chunks), \
             patch("backend.rag_pipeline._sync_if_needed",   return_value=None), \
             patch("backend.rag_pipeline.retrieve",          return_value=mock_retrieve_results), \
             patch("backend.rag_pipeline.generate_answer",   return_value="Answer."):
            result = rag_pipeline("deadlock?")
        assert "os_notes.pdf" in result["source_files"]

    def test_sources_are_text_previews(self, sample_chunks, mock_retrieve_results):
        from backend.rag_pipeline import rag_pipeline
        with patch("backend.rag_pipeline.load_chunks",       return_value=sample_chunks), \
             patch("backend.rag_pipeline._sync_if_needed",   return_value=None), \
             patch("backend.rag_pipeline.retrieve",          return_value=mock_retrieve_results), \
             patch("backend.rag_pipeline.generate_answer",   return_value="Answer."):
            result = rag_pipeline("deadlock?")
        for preview in result["sources"]:
            assert isinstance(preview, str)
            assert len(preview) <= 153  # 150 chars + "..."

    def test_long_chunks_are_truncated_in_sources(self, sample_chunks):
        from backend.rag_pipeline import rag_pipeline
        long_chunk = {**sample_chunks[0], "text": "x" * 300, "score": 0.8,
                      "semantic_score": 0.8, "keyword_score": 0.5}
        with patch("backend.rag_pipeline.load_chunks",       return_value=sample_chunks), \
             patch("backend.rag_pipeline._sync_if_needed",   return_value=None), \
             patch("backend.rag_pipeline.retrieve",          return_value=[long_chunk]), \
             patch("backend.rag_pipeline.generate_answer",   return_value="Answer."):
            result = rag_pipeline("deadlock?")
        assert result["sources"][0].endswith("...")
        assert len(result["sources"][0]) == 153
