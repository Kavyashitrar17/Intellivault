
import re
import logging
from typing import List, Dict

from backend.config import settings

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS  = 1800
MAX_CONTEXT_CHUNKS = 3
MAX_ANSWER_CHARS   = 600
MIN_ANSWER_LENGTH  = 10

_NOT_FOUND_PHRASES = [
    "not found in", "not mentioned", "no information",
    "cannot find", "does not contain", "not provided",
    "not available", "no relevant", "i don't know",
    "i do not know", "unable to find", "not_in_context",
    "not_found",
]

_PREAMBLE_PATTERNS = [
    r"^based on (the )?(provided )?(context|document[s]?)[,.]?\s*",
    r"^according to (the )?(provided )?(context|document[s]?)[,.]?\s*",
    r"^from (the )?(provided )?(context|document[s]?)[,.]?\s*",
    r"^the (context|document[s]?) (state[s]?|mention[s]?|indicate[s]?)[,.]?\s*",
    r"^(here is|here are) (a )?(summary|the answer)[:.]\s*",
    r"^(sure[,!]?|certainly[,!]?|of course[,!]?|great[,!]?|absolutely[,!]?)\s*",
]

# -------------------------------------------------------
# Groq availability check at import time
# -------------------------------------------------------
try:
    from groq import Groq as _GroqClient
    _GROQ_AVAILABLE = True
    logger.info("[QA] ✅ groq package available.")
except ImportError:
    _GroqClient = None
    _GROQ_AVAILABLE = False
    logger.error("[QA] ❌ groq package NOT installed. Run: pip install groq")

# Flan-T5 singleton (lazy)
_flan_pipeline = None


def _get_flan():
    global _flan_pipeline
    if _flan_pipeline is not None:
        return _flan_pipeline
    try:
        from transformers import pipeline
        logger.info("[QA] Loading Flan-T5...")
        _flan_pipeline = pipeline(
            "text2text-generation",
            model=settings.FLAN_MODEL_NAME,
            max_new_tokens=200,
            do_sample=False,
        )
        logger.info("[QA] Flan-T5 ready.")
    except Exception as e:
        logger.warning(f"[QA] Flan-T5 unavailable: {e}")
        _flan_pipeline = "unavailable"
    return _flan_pipeline


# -------------------------------------------------------
# Chunk text cleaning
# -------------------------------------------------------

def _clean_chunk_text(text: str) -> str:
    """
    Strip document title lines, page headers, and numbered section headers
    before sending chunk text to the LLM or building source previews.

    Example — BEFORE:
        "IntelliVault Model Training Notes 1. Overview IntelliVault is..."
    AFTER:
        "IntelliVault is a secure document intelligence system that uses..."
    """
    lines   = text.splitlines()
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            if cleaned:        # keep internal blank lines, drop leading ones
                cleaned.append("")
            continue
        # Short title-case line with no sentence-ending punctuation → header
        is_short_title  = len(s) < 70 and s.istitle() and not s.endswith(".")
        # "1. Overview" / "2. Data Collection"
        is_numbered_hdr = bool(re.match(r"^\d+\.\s+[A-Z][a-zA-Z ]{2,30}$", s))
        # Known boilerplate patterns
        is_boilerplate  = any(kw in s.lower() for kw in [
            "training notes", "table of contents", "model training notes",
            "page ", "section ", "appendix",
        ])
        if is_short_title or is_numbered_hdr or is_boilerplate:
            continue
        cleaned.append(line)

    result = "\n".join(cleaned).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


# -------------------------------------------------------
# Prompts
# -------------------------------------------------------

def _system_prompt() -> str:
    return (
        "You are a document Q&A assistant. "
        "You answer questions STRICTLY using the provided context.\n\n"
        "ABSOLUTE RULES — follow every one:\n"
        "1. Use ONLY information from the context. "
        "   Do NOT add external knowledge, facts, or definitions.\n"
        "2. Do NOT copy sentences word-for-word from the context. "
        "   Paraphrase and summarize.\n"
        "3. Keep answers SHORT: max 3 sentences OR a bullet list of max 5 items.\n"
        "4. When listing features, steps, or multiple items → use bullet points "
        "   starting with '- ' on separate lines.\n"
        "5. NEVER start with 'Based on', 'According to', 'The context says', "
        "   or any similar preamble.\n"
        "6. If the answer is not in the context → reply with exactly: NOT_FOUND\n"
        "7. No filler, no padding, no markdown headers (#, ##).\n"
        "8. No hallucination. If unsure → say NOT_FOUND."
    )


def _user_prompt(query: str, context: str) -> str:
    return f"""
CONTEXT (use only this):
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer ONLY from context
- DO NOT copy sentences
- Rewrite in your own words
- Keep answer SHORT
- Use bullet points ONLY
- Max 3–4 bullets
- Each bullet = one idea
- NO paragraph
- NO extra explanation
- If not found → return: NOT_FOUND

ANSWER:
"""


# -------------------------------------------------------
# Main entry point
# -------------------------------------------------------

def generate_answer(query: str, chunks: List[Dict]) -> str:
    if not chunks:
        return "No relevant content found. Please upload a document first."

    clean_chunks = _deduplicate_chunks(chunks[:MAX_CONTEXT_CHUNKS])
    context = _build_context(clean_chunks, MAX_CONTEXT_CHARS)

    print("\n" + "="*50)
    print("🔥 USING GROQ ONLY")
    print(f"Query: {query}")
    print("="*50)

    if not (_GROQ_AVAILABLE and settings.GROQ_API_KEY):
        raise RuntimeError("❌ GROQ NOT AVAILABLE OR API KEY MISSING")

    answer = _answer_groq(query, context)

    if not answer or _is_empty_or_not_found(answer):
        raise RuntimeError("❌ GROQ FAILED OR RETURNED EMPTY")

    return _clean_output(answer)
# -------------------------------------------------------
# LLM backends
# -------------------------------------------------------

def _answer_groq(query: str, context: str) -> str:
    print("🚀 Calling Groq API...")

    client = _GroqClient(api_key=settings.GROQ_API_KEY.strip())

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ FIXED
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(query, context)},
            ],
            max_tokens=300,
            temperature=0.0,
        )

        answer = response.choices[0].message.content.strip()
        print("✅ Groq response received")

        return answer

    except Exception as e:
        print("❌ GROQ ERROR:", str(e))
        raise RuntimeError("Groq API failed")

def _answer_openai(query: str, context: str) -> str:
    """Call OpenAI. Raises on ANY error — caller handles it."""
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY.strip())
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user",   "content": _user_prompt(query, context)},
        ],
        max_tokens=300,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def _answer_flan(query: str, context: str, pipeline_obj) -> str:
    """Flan-T5 local inference. Raises on error — caller handles it."""
    prompt = (
        "Summarize the answer to the question in 2 sentences "
        "using ONLY the context. Do not copy text. Do not add outside knowledge.\n\n"
        f"Context: {context[:800]}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    result = pipeline_obj(prompt, max_new_tokens=150, do_sample=False)
    return result[0]["generated_text"].strip()


# -------------------------------------------------------
# Context building
# -------------------------------------------------------

def _deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    selected, sigs = [], []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        sig = _ngram_sig(text)
        if any(_jaccard(sig, s) > 0.6 for s in sigs):
            logger.debug(f"[QA] Deduped chunk: {text[:50]}")
            continue
        selected.append(chunk)
        sigs.append(sig)
    logger.info(f"[QA] {len(selected)}/{len(chunks)} chunks after dedup.")
    return selected


def _build_context(chunks: List[Dict], max_chars: int) -> str:
    """
    Build numbered context with cleaned text.
    Sections separated by --- so LLM can distinguish chunk boundaries.
    """
    parts, total = [], 0
    for i, chunk in enumerate(chunks, 1):
        raw  = chunk.get("text", "").strip()
        text = _clean_chunk_text(raw)
        if not text:
            text = raw
        entry = f"[Source {i}]\n{text}"
        if total + len(entry) > max_chars:
            rem = max_chars - total
            if rem > 80:
                parts.append(entry[:rem].rstrip() + "...")
            break
        parts.append(entry)
        total += len(entry) + 2
    return "\n\n---\n\n".join(parts)


def _ngram_sig(text: str, n: int = 5) -> set:
    t = re.sub(r"\s+", " ", text.lower().strip())
    return {t[i:i+n] for i in range(len(t) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# -------------------------------------------------------
# Output cleaning  (FIX: converts \n → real newlines)
# -------------------------------------------------------

def _clean_output(text: str) -> str:
    text = text.replace("\\n", "\n").strip()

    # If not bullet → convert
    if "-" not in text:
        sentences = re.split(r"[.]", text)
        bullets = [f"- {s.strip()}" for s in sentences if s.strip()]
        text = "\n".join(bullets[:4])

    # remove duplicates
    seen = set()
    final = []
    for line in text.splitlines():
        key = line.strip().lower()
        if key and key not in seen:
            seen.add(key)
            final.append(line)

    return "\n".join(final[:4])


def _is_empty_or_not_found(text: str) -> bool:
    if not text or len(text.strip()) < MIN_ANSWER_LENGTH:
        return True
    lower = text.lower()
    return any(p in lower for p in _NOT_FOUND_PHRASES)


# -------------------------------------------------------
# Extractive fallback
# -------------------------------------------------------

_STOPWORDS = {
    "a","an","and","are","as","at","be","been","but","by","can","did","do","does",
    "for","from","had","has","have","how","i","if","in","into","is","it","its",
    "may","no","not","of","on","or","our","out","should","so","than","that","the",
    "their","then","there","these","they","this","those","to","was","we","were",
    "what","when","where","which","who","will","with","you",
}


def _is_header_line(s: str) -> bool:
    s = s.strip()
    return (
        len(s) < 70
        and not s.endswith(".")
        and (s.istitle() or bool(re.match(r"^\d+\.\s+[A-Z]", s)))
    )


def _extractive_answer(query: str, chunks: List[Dict]) -> str:
    query_tokens = set(_tok(query))
    key_terms    = {t for t in query_tokens if len(t) >= 3 and t not in _STOPWORDS}
    scored       = []

    for rank, chunk in enumerate(chunks):
        text  = _clean_chunk_text(chunk.get("text", ""))
        sents = _split_sents(text)
        for pos, sent in enumerate(sents):
            if len(sent) < 20 or _is_header_line(sent):
                continue
            s_tok = set(_tok(sent))
            if key_terms and not (s_tok & key_terms):
                continue
            kw    = len(query_tokens & s_tok) / len(query_tokens) if query_tokens else 0
            kr    = len(key_terms & s_tok)    / len(key_terms)    if key_terms   else 0
            score = max(0.65 * kr + 0.35 * kw + 0.1 * (1 - pos / max(len(sents), 1)), 0.0)
            scored.append((score, sent))

    if not scored:
        sents      = _split_sents(_clean_chunk_text(chunks[0].get("text","") if chunks else ""))
        non_header = [s for s in sents if not _is_header_line(s) and len(s) > 20]
        return " ".join(non_header[:2]) or "Answer not found in documents."

    scored.sort(key=lambda x: x[0], reverse=True)
    selected, seen_texts = [], []
    for _, sent in scored:
        if len(selected) >= 2:
            break
        if not _near_dup(sent, seen_texts):
            selected.append(sent)
            seen_texts.append(sent)

    return " ".join(selected) or "Answer not found in documents."


def _split_sents(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', text) if s.strip()]


def _tok(text: str) -> List[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _near_dup(candidate: str, selected: List[str], t: float = 0.55) -> bool:
    c = set(_tok(candidate))
    for s in selected:
        sv = set(_tok(s))
        u  = c | sv
        if u and len(c & sv) / len(u) >= t:
            return True
    return False


# -------------------------------------------------------
# Confidence scorer
# -------------------------------------------------------

def score_confidence(top_chunks: List[Dict], answer: str) -> str:
    if not top_chunks or not answer:
        return "none"
    if _is_empty_or_not_found(answer):
        return "none"
    top_score  = float(top_chunks[0].get("score", 0) or 0)
    penalty    = (0.15 if top_chunks[0].get("fallback") else 0) + \
                 (0.10 if len(answer) < 40 else 0)
    effective  = top_score - penalty
    if effective >= settings.CONFIDENCE_HIGH:
        return "high"
    if effective >= settings.CONFIDENCE_MEDIUM:
        return "medium"
    return "low"