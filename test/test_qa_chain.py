"""
tests/test_qa_chain.py
----------------------
Tests for backend/llm/qa_chain.py

The real LLM (Flan-T5) is mocked so tests run instantly without
downloading or loading the model.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def chunks_with_answer():
    return [
        {
            "chunk_id": 0,
            "source":   "os_notes.pdf",
            "text":     (
                "A deadlock is a situation where two or more processes are unable "
                "to proceed because each is waiting for the other to release a resource."
            ),
            "score": 0.75,
        },
        {
            "chunk_id": 1,
            "source":   "os_notes.pdf",
            "text":     (
                "Deadlocks can be prevented by resource ordering, timeouts, "
                "or the banker's algorithm."
            ),
            "score": 0.55,
        },
    ]


@pytest.fixture
def chunks_without_answer():
    return [
        {
            "chunk_id": 0,
            "source":   "cooking.pdf",
            "text":     "Boil the pasta for 10 minutes. Add salt to taste.",
            "score":    0.20,
        }
    ]


class TestGenerateAnswer:

    def test_returns_string(self, chunks_with_answer):
        from backend.llm.qa_chain import generate_answer
        with patch("backend.llm.qa_chain._get_llm", return_value="extractive"):
            result = generate_answer("What is a deadlock?", chunks_with_answer)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_chunks_returns_fallback_message(self):
        from backend.llm.qa_chain import generate_answer
        result = generate_answer("What is a deadlock?", [])
        assert "upload" in result.lower() or "no relevant" in result.lower()

    def test_flan_answer_used_when_non_empty(self, chunks_with_answer):
        from backend.llm.qa_chain import generate_answer
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "A deadlock is a waiting state."}]
        with patch("backend.llm.qa_chain._get_llm", return_value=mock_pipeline):
            result = generate_answer("What is a deadlock?", chunks_with_answer)
        assert "deadlock" in result.lower()

    def test_falls_back_to_extractive_when_flan_returns_empty(self, chunks_with_answer):
        from backend.llm.qa_chain import generate_answer
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": ""}]
        with patch("backend.llm.qa_chain._get_llm",       return_value=mock_pipeline), \
             patch("backend.llm.qa_chain._extractive_answer", return_value="Extractive answer.") as mock_ext:
            result = generate_answer("What is a deadlock?", chunks_with_answer)
        mock_ext.assert_called_once()

    def test_openai_path_called_when_configured(self, chunks_with_answer):
        from backend.llm.qa_chain import generate_answer
        from backend import config
        with patch.object(config.settings, "LLM_PROVIDER",   "openai"), \
             patch.object(config.settings, "OPENAI_API_KEY",  "sk-test"), \
             patch("backend.llm.qa_chain._answer_openai", return_value="OpenAI answer.") as mock_oai:
            result = generate_answer("What is a deadlock?", chunks_with_answer)
        mock_oai.assert_called_once()
        assert result == "OpenAI answer."

    def test_groq_path_called_when_configured(self, chunks_with_answer):
        from backend.llm.qa_chain import generate_answer
        from backend import config
        with patch.object(config.settings, "LLM_PROVIDER",  "groq"), \
             patch.object(config.settings, "GROQ_API_KEY",   "gsk-test"), \
             patch("backend.llm.qa_chain._answer_groq", return_value="Groq answer.") as mock_groq:
            result = generate_answer("What is a deadlock?", chunks_with_answer)
        mock_groq.assert_called_once()
        assert result == "Groq answer."


class TestExtractiveAnswer:

    def test_finds_answer_in_matching_chunk(self, chunks_with_answer):
        from backend.llm.qa_chain import _extractive_answer
        result = _extractive_answer("What is a deadlock?", chunks_with_answer)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_not_found_message_for_unrelated_chunks(self, chunks_without_answer):
        from backend.llm.qa_chain import _extractive_answer
        result = _extractive_answer("What is a deadlock?", chunks_without_answer)
        assert "not found" in result.lower()

    def test_answer_contains_relevant_terms(self, chunks_with_answer):
        from backend.llm.qa_chain import _extractive_answer
        result = _extractive_answer("What is a deadlock?", chunks_with_answer)
        assert "deadlock" in result.lower()

    def test_empty_chunks_returns_fallback(self):
        from backend.llm.qa_chain import _extractive_answer
        result = _extractive_answer("What?", [])
        assert "no relevant" in result.lower() or "not found" in result.lower()


class TestBuildContext:

    def test_context_truncated_to_max_chars(self, chunks_with_answer):
        from backend.llm.qa_chain import _build_context
        context = _build_context(chunks_with_answer, max_chars=50)
        assert len(context) <= 50 + 10  # small tolerance for separator

    def test_context_is_string(self, chunks_with_answer):
        from backend.llm.qa_chain import _build_context
        assert isinstance(_build_context(chunks_with_answer), str)

    def test_empty_chunks_returns_empty_string(self):
        from backend.llm.qa_chain import _build_context
        assert _build_context([]) == ""
