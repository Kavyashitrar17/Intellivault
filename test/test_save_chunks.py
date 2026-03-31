"""
tests/test_save_chunks.py
-------------------------
Tests for backend/ingestion/save_chunks.py
"""

import os
import json
import pytest
from backend.ingestion.save_chunks import save_chunks, load_chunks, count_chunks


class TestSaveChunks:

    def test_saves_and_returns_total_count(self, sample_chunks):
        total = save_chunks(sample_chunks)
        assert total == len(sample_chunks)

    def test_appends_on_second_call(self, sample_chunks):
        save_chunks(sample_chunks)
        total = save_chunks(sample_chunks)
        assert total == len(sample_chunks) * 2

    def test_chunk_ids_are_sequential(self, sample_chunks):
        save_chunks(sample_chunks)
        chunks = load_chunks()
        ids = [c["chunk_id"] for c in chunks]
        # chunk_ids are re-numbered as integers 0, 1, 2, ...
        assert ids == list(range(len(sample_chunks)))

    def test_ids_continue_after_append(self, sample_chunks):
        save_chunks(sample_chunks)
        save_chunks(sample_chunks)
        chunks = load_chunks()
        expected = list(range(len(sample_chunks) * 2))
        assert [c["chunk_id"] for c in chunks] == expected

    def test_saves_as_valid_json(self, sample_chunks):
        from backend.config import settings
        save_chunks(sample_chunks)
        with open(settings.CHUNKS_PATH, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_source_preserved(self, sample_chunks):
        save_chunks(sample_chunks)
        loaded = load_chunks()
        for chunk in loaded:
            assert chunk["source"] == "os_notes.pdf"


class TestLoadChunks:

    def test_returns_empty_when_no_file(self):
        chunks = load_chunks()
        assert chunks == []

    def test_returns_saved_chunks(self, sample_chunks):
        save_chunks(sample_chunks)
        loaded = load_chunks()
        assert len(loaded) == len(sample_chunks)

    def test_each_chunk_has_required_keys(self, sample_chunks):
        save_chunks(sample_chunks)
        for chunk in load_chunks():
            assert "chunk_id" in chunk
            assert "source"   in chunk
            assert "text"     in chunk


class TestCountChunks:

    def test_zero_when_no_file(self):
        assert count_chunks() == 0

    def test_correct_count_after_save(self, sample_chunks):
        save_chunks(sample_chunks)
        assert count_chunks() == len(sample_chunks)

    def test_count_after_append(self, sample_chunks):
        save_chunks(sample_chunks)
        save_chunks(sample_chunks)
        assert count_chunks() == len(sample_chunks) * 2
