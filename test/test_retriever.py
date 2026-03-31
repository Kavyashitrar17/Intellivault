"""
tests/test_retriever.py
-----------------------
Tests for backend/retrieval/retriever.py
"""

import numpy as np
import pytest
from unittest.mock import patch
from backend.retrieval.vector_store import reset_vector_store


@pytest.fixture(autouse=True)
def reset_store():
    reset_vector_store()
    yield
    reset_vector_store()


class TestRetrieve:

    def test_returns_list(self, sample_chunks, loaded_store):
        from backend.retrieval.retriever import retrieve
        results = retrieve("What is a deadlock?", sample_chunks, k=2, store=loaded_store)
        assert isinstance(results, list)

    def test_returns_at_most_k_results(self, sample_chunks, loaded_store):
        from backend.retrieval.retriever import retrieve
        results = retrieve("deadlock", sample_chunks, k=2, store=loaded_store)
        assert len(results) <= 2

    def test_each_result_has_score(self, sample_chunks, loaded_store):
        from backend.retrieval.retriever import retrieve
        results = retrieve("deadlock", sample_chunks, k=3, store=loaded_store)
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_results_sorted_best_first(self, sample_chunks, loaded_store):
        from backend.retrieval.retriever import retrieve
        results = retrieve("deadlock prevention", sample_chunks, k=3, store=loaded_store)
        if len(results) < 2:
            pytest.skip("Not enough results to check ordering")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_each_result_has_text_and_source(self, sample_chunks, loaded_store):
        from backend.retrieval.retriever import retrieve
        results = retrieve("deadlock", sample_chunks, k=3, store=loaded_store)
        for r in results:
            assert "text"   in r
            assert "source" in r

    def test_empty_chunks_returns_empty(self, loaded_store):
        from backend.retrieval.retriever import retrieve
        results = retrieve("deadlock", [], k=3, store=loaded_store)
        assert results == []

    def test_k_zero_returns_empty(self, sample_chunks, loaded_store):
        from backend.retrieval.retriever import retrieve
        results = retrieve("deadlock", sample_chunks, k=0, store=loaded_store)
        assert results == []

    def test_empty_store_returns_empty(self, sample_chunks, tmp_data_dir):
        from backend.retrieval.retriever import retrieve
        from backend.retrieval.vector_store import VectorStore
        import os
        empty_store = VectorStore(index_path=os.path.join(tmp_data_dir, "empty_ret.index"))
        results = retrieve("deadlock", sample_chunks, k=3, store=empty_store)
        assert results == []

    def test_score_between_zero_and_one(self, sample_chunks, loaded_store):
        from backend.retrieval.retriever import retrieve
        results = retrieve("deadlock", sample_chunks, k=3, store=loaded_store)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_semantic_and_keyword_scores_present(self, sample_chunks, loaded_store):
        from backend.retrieval.retriever import retrieve
        results = retrieve("deadlock", sample_chunks, k=3, store=loaded_store)
        for r in results:
            assert "semantic_score" in r
            assert "keyword_score"  in r


class TestKeywordScore:

    def test_full_overlap_returns_one(self):
        from backend.retrieval.retriever import _keyword_score, _tokenize
        tokens = _tokenize("deadlock process resource")
        score  = _keyword_score(tokens, "deadlock process resource")
        assert score == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self):
        from backend.retrieval.retriever import _keyword_score, _tokenize
        tokens = _tokenize("banana apple")
        score  = _keyword_score(tokens, "deadlock mutex semaphore")
        assert score == 0.0

    def test_empty_query_returns_zero(self):
        from backend.retrieval.retriever import _keyword_score
        assert _keyword_score([], "any text") == 0.0

    def test_partial_overlap(self):
        from backend.retrieval.retriever import _keyword_score, _tokenize
        tokens = _tokenize("deadlock process")
        score  = _keyword_score(tokens, "deadlock prevention technique")
        assert 0.0 < score < 1.0
