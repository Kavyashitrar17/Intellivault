"""
qa_chain.py
-----------
Answer generation with a REAL LLM (Flan-T5) + extractive fallback.

BIGGEST UPGRADE IN THE PROJECT:
  The original used pure keyword scoring to pick sentences. That's not
  actually a language model — it's a search algorithm. This version uses
  google/flan-t5-base, a free, CPU-friendly seq2seq model (~250 MB) that:
    - Understands natural language questions
    - Can paraphrase and synthesise answers (not just copy sentences)
    - Runs on CPU in ~1–2 seconds per query
    - Requires NO API key

  If you later want GPT-4 quality, set LLM_PROVIDER=openai in .env
  and add your OPENAI_API_KEY. The same interface works for both.

PIPELINE:
  1. Build a context string from the top retrieved chunks
  2. Format a QA prompt: "Context: ... Question: ... Answer:"
  3. Run through Flan-T5 (or OpenAI if configured)
  4. If LLM returns an empty/unusable answer, fall back to extractive QA

Install:  pip install transformers accelerate sentencepiece
"""

import re
import logging
from typing import List, Dict

from backend.config import settings

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# LLM singleton (lazy)
# -------------------------------------------------------
_llm_pipeline = None


def _get_llm():
    """Load Flan-T5 pipeline once, reuse forever."""
    global _llm_pipeline
    if _llm_pipeline is not None:
        return _llm_pipeline

    if settings.LLM_PROVIDER == "openai":
        return None  # OpenAI path doesn't need a local pipeline

    if settings.LLM_PROVIDER == "groq":
        return None  # Groq path doesn't need a local pipeline

    # Default: flan-t5-base
    try:
        from transformers import pipeline
        logger.info(f"[QA] Loading '{settings.FLAN_MODEL_NAME}' (first run may download ~250 MB)...")
        _llm_pipeline = pipeline(
            "text2text-generation",
            model=settings.FLAN_MODEL_NAME,
            max_new_tokens=200,
            do_sample=False,       # deterministic
        )
        logger.info("[QA] Flan-T5 model ready.")
    except ImportError:
        logger.warning(
            "[QA] transformers not installed — falling back to extractive QA. "
            "Run: pip install transformers sentencepiece"
        )
        _llm_pipeline = "extractive"
    except Exception as e:
        logger.error(f"[QA] Failed to load Flan-T5: {e} — using extractive fallback.")
        _llm_pipeline = "extractive"

    return _llm_pipeline


# -------------------------------------------------------
# Public interface
# -------------------------------------------------------

def generate_answer(query: str, chunks: List[Dict]) -> str:
    """
    Generate an answer for `query` using the top retrieved chunks.

    Args:
        query:  The user's question.
        chunks: Ranked list of chunk dicts (from retriever.py).

    Returns:
        Answer string.
    """
    if not chunks:
        return "No relevant content found. Please upload a document first."

    provider = settings.LLM_PROVIDER

    if provider == "openai" and settings.OPENAI_API_KEY:
        return _answer_openai(query, chunks)

    if provider == "groq" and settings.GROQ_API_KEY:
        return _answer_groq(query, chunks)

    # Default: Flan-T5 local model
    pipeline_obj = _get_llm()
    if pipeline_obj and pipeline_obj != "extractive":
        answer = _answer_flan(query, chunks, pipeline_obj)
        if answer and len(answer.strip()) > 5:
            return answer.strip()
        logger.info("[QA] Flan-T5 returned empty answer — using extractive fallback.")

    # Final fallback: improved extractive QA
    return _extractive_answer(query, chunks)


# -------------------------------------------------------
# LLM backends
# -------------------------------------------------------

def _build_context(chunks: List[Dict], max_chars: int = 1500) -> str:
    """Concatenate top chunks into a context string (truncated to fit model)."""
    context_parts = []
    total = 0
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                context_parts.append(text[:remaining])
            break
        context_parts.append(text)
        total += len(text)
    return "\n\n".join(context_parts)


def _answer_flan(query: str, chunks: List[Dict], pipeline_obj) -> str:
    """Use Flan-T5 for answer generation."""
    context = _build_context(chunks, max_chars=1200)
    prompt = (
        f"Answer the question based only on the context below. "
        f"If the answer is not in the context, say 'Not found in documents'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    try:
        result = pipeline_obj(prompt, max_new_tokens=150, do_sample=False)
        return result[0]["generated_text"].strip()
    except Exception as e:
        logger.error(f"[QA] Flan-T5 inference failed: {e}")
        return ""


def _answer_openai(query: str, chunks: List[Dict]) -> str:
    """Use OpenAI API for answer generation."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        context = _build_context(chunks, max_chars=3000)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document Q&A assistant. Answer only from the "
                        "provided context. If the answer is not there, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
            max_tokens=300,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[QA] OpenAI call failed: {e}")
        return _extractive_answer(query, chunks)


def _answer_groq(query: str, chunks: List[Dict]) -> str:
    """Use Groq API (free tier) for fast LLM answers."""
    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        context = _build_context(chunks, max_chars=3000)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document Q&A assistant. Answer only from the "
                        "provided context. If the answer is not there, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
            max_tokens=300,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[QA] Groq call failed: {e}")
        return _extractive_answer(query, chunks)


# -------------------------------------------------------
# Extractive fallback (improved version of original)
# -------------------------------------------------------

_STOPWORDS = {
    "a","an","and","are","as","at","be","been","being","but","by","can","could",
    "did","do","does","doing","for","from","had","has","have","having","how","i",
    "if","in","into","is","it","its","may","might","more","most","must","no","not",
    "of","on","or","our","out","should","so","such","than","that","the","their",
    "then","there","these","they","this","those","to","was","we","were","what",
    "when","where","which","who","will","with","would","you","your",
}

TOP_SENTENCES    = 2
MIN_SENT_LENGTH  = 15
MIN_SCORE_THRESHOLD = 0.45  # slightly lower than original's 0.55


def _extractive_answer(query: str, chunks: List[Dict]) -> str:
    query_tokens  = set(_tokenize(query))
    key_terms     = {t for t in query_tokens if len(t) >= 3 and t not in _STOPWORDS}
    all_scored    = []

    for rank, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if not text:
            continue
        sentences = _split_sentences(text)
        for pos, sent in enumerate(sentences):
            if len(sent) < MIN_SENT_LENGTH:
                continue
            sent_tokens = set(_tokenize(sent))
            if key_terms and not (sent_tokens & key_terms):
                continue
            score = _score(sent_tokens, query_tokens, key_terms, pos, len(sentences), rank)
            all_scored.append((score, sent, rank, pos))

    if not all_scored:
        return "Answer not found in documents."

    all_scored.sort(key=lambda x: x[0], reverse=True)
    best_score = all_scored[0][0]

    if best_score < MIN_SCORE_THRESHOLD:
        return "Answer not found in documents."

    selected, selected_texts = [], []
    for score, sent, rank, pos in all_scored:
        if len(selected) >= TOP_SENTENCES:
            break
        if not _is_near_duplicate(sent, selected_texts):
            selected.append((sent, rank, pos))
            selected_texts.append(sent)

    if not selected:
        selected = [(all_scored[0][1], all_scored[0][2], all_scored[0][3])]

    selected.sort(key=lambda x: (x[1], x[2]))
    return " ".join(s for s, _, _ in selected).strip()


def _split_sentences(text: str) -> List[str]:
    raw = re.split(r'(?<=[.!?])\s+|\n', text)
    return [s.strip() for s in raw if s.strip()]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _score(sent_tokens, query_tokens, key_terms, pos, total, rank) -> float:
    kw_score     = len(query_tokens & sent_tokens) / len(query_tokens) if query_tokens else 0
    key_ratio    = len(key_terms & sent_tokens) / len(key_terms) if key_terms else 0
    pos_bonus    = 0.2 * (1 - pos / total) if total else 0
    word_count   = len(sent_tokens)
    len_bonus    = 0.1 if 10 <= word_count <= 40 else (0.05 if word_count > 40 else 0)
    rank_penalty = 0.05 * rank
    return max(0.7 * key_ratio + 0.3 * kw_score + pos_bonus + len_bonus - rank_penalty, 0.0)


def _is_near_duplicate(candidate: str, selected: List[str], threshold: float = 0.6) -> bool:
    cand_tok = set(_tokenize(candidate))
    for existing in selected:
        ext_tok = set(_tokenize(existing))
        union   = cand_tok | ext_tok
        if union and len(cand_tok & ext_tok) / len(union) >= threshold:
            return True
    return False