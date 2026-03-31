# IntelliVault — Test Suite

## Setup

```bash
pip install pytest pytest-cov httpx
```

## Run all tests

```bash
pytest tests/ -v
```

## Run with coverage report

```bash
pytest tests/ -v --cov=backend --cov-report=term-missing
```

## Run a single file

```bash
pytest tests/test_chunker.py -v
pytest tests/test_api.py -v
```

## Run a single test

```bash
pytest tests/test_vector_store.py::TestSearch::test_search_self_is_top_result -v
```

## Skip slow tests (embedding model loads)

```bash
pytest tests/ -v -m "not slow"
```

---

## What's tested

| File | Tests | What it covers |
|---|---|---|
| `test_chunker.py` | 12 | Overlap, min words, empty input, chunk IDs |
| `test_embedder.py` | 10 | Shape, dtype, normalization, singleton, similarity |
| `test_vector_store.py` | 17 | Add, search, save/load, singleton, rebuild |
| `test_retriever.py` | 12 | Hybrid scoring, empty edge cases, sort order |
| `test_save_chunks.py` | 11 | JSON save/load, ID sequencing, count |
| `test_rag_pipeline.py` | 9 | Confidence levels, no-data paths, source extraction |
| `test_qa_chain.py` | 13 | Flan/OpenAI/Groq routing, extractive fallback, context |
| `test_api.py` | 20 | All endpoints, auth, validation, error codes |

**Total: ~104 tests**

---

## Design decisions

- All tests use `tmp_data_dir` — they never read or write your real
  `data/` folder. Safe to run at any time.
- The embedding model and LLM are mocked in API and pipeline tests
  so the suite runs in seconds, not minutes.
- `conftest.py` patches `settings` paths automatically via `autouse=True`,
  so individual test files don't need to repeat this setup.
