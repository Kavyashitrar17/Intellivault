"""
tests/test_vector_store.py
--------------------------
Tests for backend/retrieval/vector_store.py
"""

import os
import numpy as np
import pytest
from backend.retrieval.vector_store import VectorStore, get_vector_store, reset_vector_store


@pytest.fixture(autouse=True)
def clear_singleton():
    """Reset singleton before and after every test in this file."""
    reset_vector_store()
    yield
    reset_vector_store()


class TestVectorStoreInit:

    def test_new_store_starts_empty(self, tmp_data_dir):
        path = os.path.join(tmp_data_dir, "test_new.index")
        store = VectorStore(index_path=path)
        assert store.total_vectors == 0

    def test_loads_existing_index(self, tmp_data_dir, sample_embeddings):
        path = os.path.join(tmp_data_dir, "test_load.index")
        s1 = VectorStore(index_path=path)
        s1.add(sample_embeddings)
        s1.save()

        s2 = VectorStore(index_path=path)
        assert s2.total_vectors == len(sample_embeddings)

    def test_backwards_compat_int_as_first_arg(self, tmp_data_dir):
        """VectorStore(384) should not crash (old calling convention)."""
        store = VectorStore(384)
        assert store.dimension == 384


class TestAdd:

    def test_add_increases_vector_count(self, tmp_data_dir, sample_embeddings):
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "add.index"))
        store.add(sample_embeddings)
        assert store.total_vectors == len(sample_embeddings)

    def test_add_twice_accumulates(self, tmp_data_dir, sample_embeddings):
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "add2.index"))
        store.add(sample_embeddings)
        store.add(sample_embeddings)
        assert store.total_vectors == len(sample_embeddings) * 2

    def test_add_empty_does_nothing(self, tmp_data_dir):
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "empty.index"))
        store.add(np.array([]))
        assert store.total_vectors == 0

    def test_add_wrong_dimension_raises(self, tmp_data_dir):
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "dim.index"))
        bad = np.random.rand(3, 128).astype("float32")
        with pytest.raises(ValueError, match="dimension"):
            store.add(bad)


class TestSearch:

    def test_search_returns_correct_types(self, tmp_data_dir, sample_embeddings):
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "search.index"))
        store.add(sample_embeddings)
        indices, scores = store.search(sample_embeddings[0:1], k=2)
        assert isinstance(indices, list)
        assert isinstance(scores,  list)

    def test_search_returns_k_results(self, tmp_data_dir, sample_embeddings):
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "k.index"))
        store.add(sample_embeddings)
        indices, scores = store.search(sample_embeddings[0:1], k=2)
        assert len(indices) == 2
        assert len(scores)  == 2

    def test_search_self_is_top_result(self, tmp_data_dir, sample_embeddings):
        """Searching with vector[i] should return index i as the top hit."""
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "self.index"))
        store.add(sample_embeddings)
        indices, scores = store.search(sample_embeddings[1:2], k=1)
        assert indices[0] == 1

    def test_search_scores_are_descending(self, tmp_data_dir, sample_embeddings):
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "desc.index"))
        store.add(sample_embeddings)
        _, scores = store.search(sample_embeddings[0:1], k=3)
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_index_returns_empty(self, tmp_data_dir, sample_embeddings):
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "empty_s.index"))
        indices, scores = store.search(sample_embeddings[0:1], k=3)
        assert indices == []
        assert scores  == []

    def test_search_k_zero_returns_empty(self, tmp_data_dir, sample_embeddings):
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "k0.index"))
        store.add(sample_embeddings)
        indices, scores = store.search(sample_embeddings[0:1], k=0)
        assert indices == []

    def test_search_1d_query_is_accepted(self, tmp_data_dir, sample_embeddings):
        """A (384,) shaped query should not crash — auto-reshaped to (1, 384)."""
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "1d.index"))
        store.add(sample_embeddings)
        flat_query = sample_embeddings[0]   # shape (384,)
        indices, scores = store.search(flat_query, k=1)
        assert len(indices) == 1


class TestSingleton:

    def test_get_vector_store_same_instance(self, tmp_data_dir):
        s1 = get_vector_store()
        s2 = get_vector_store()
        assert s1 is s2

    def test_reset_clears_singleton(self, tmp_data_dir):
        s1 = get_vector_store()
        reset_vector_store()
        s2 = get_vector_store()
        assert s1 is not s2


class TestRebuild:

    def test_rebuild_replaces_index(self, tmp_data_dir, sample_embeddings):
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "rebuild.index"))
        store.add(sample_embeddings)
        assert store.total_vectors == 3

        new_emb = sample_embeddings[:2]
        store.rebuild_from_embeddings(new_emb)
        assert store.total_vectors == 2

    def test_rebuild_empty_raises(self, tmp_data_dir):
        store = VectorStore(index_path=os.path.join(tmp_data_dir, "rebuildE.index"))
        with pytest.raises(ValueError):
            store.rebuild_from_embeddings(np.array([]))
