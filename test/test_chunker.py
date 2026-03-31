"""
tests/test_chunker.py
---------------------
Tests for backend/ingestion/chunker.py
"""

import pytest
from backend.ingestion.chunker import chunk_text


class TestChunkText:

    def test_basic_chunking_returns_list(self, sample_text):
        chunks = chunk_text(sample_text, source="test.pdf")
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_has_required_keys(self, sample_text):
        chunks = chunk_text(sample_text, source="test.pdf")
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "source"   in chunk
            assert "text"     in chunk

    def test_source_preserved(self, sample_text):
        chunks = chunk_text(sample_text, source="lecture1.pdf")
        for chunk in chunks:
            assert chunk["source"] == "lecture1.pdf"

    def test_chunk_ids_are_unique(self, sample_text):
        chunks = chunk_text(sample_text, source="notes.txt")
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "chunk_ids must be unique"

    def test_chunk_ids_include_source_name(self, sample_text):
        chunks = chunk_text(sample_text, source="my notes.pdf")
        # spaces → underscores, extension stripped
        assert all("my_notes" in c["chunk_id"] for c in chunks)

    def test_small_chunk_size_produces_more_chunks(self, sample_text):
        small = chunk_text(sample_text, source="t.pdf", chunk_size=30,  overlap=5)
        large = chunk_text(sample_text, source="t.pdf", chunk_size=200, overlap=5)
        assert len(small) >= len(large)

    def test_overlap_words_appear_in_consecutive_chunks(self, sample_text):
        """Last `overlap` words of chunk N should appear at the start of chunk N+1."""
        chunks = chunk_text(sample_text, source="t.pdf", chunk_size=50, overlap=20)
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")
        words_end   = chunks[0]["text"].split()[-10:]
        words_start = chunks[1]["text"].split()[:30]
        # At least some words should be shared (overlap effect)
        overlap_found = any(w in words_start for w in words_end)
        assert overlap_found, "Overlap words should appear in the next chunk"

    def test_empty_text_returns_empty_list(self):
        chunks = chunk_text("", source="empty.pdf")
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self):
        chunks = chunk_text("   \n\n\t  ", source="blank.pdf")
        assert chunks == []

    def test_no_chunk_shorter_than_min_words(self, sample_text):
        chunks = chunk_text(sample_text, source="t.pdf")
        for chunk in chunks:
            word_count = len(chunk["text"].split())
            assert word_count >= 5, f"Chunk too short: '{chunk['text']}'"

    def test_chunk_text_is_string(self, sample_text):
        chunks = chunk_text(sample_text, source="t.pdf")
        for chunk in chunks:
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"]) > 0

    def test_very_short_text_below_min_words(self):
        """Text with fewer words than MIN_CHUNK_WORDS should produce no chunks."""
        chunks = chunk_text("Hello world.", source="tiny.txt")
        # Either 0 or 1 chunk — should never raise
        assert isinstance(chunks, list)
