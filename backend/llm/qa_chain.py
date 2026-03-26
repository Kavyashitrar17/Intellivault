"""
qa_chain.py
-----------
Generates an answer from retrieved chunks — no LLM API needed.

APPROACH: Improved Extractive QA
  1. Take the top retrieved chunks (already ranked by retriever.py).
  2. Split each chunk into individual sentences.
  3. Score every sentence by how well it matches the query.
  4. Return the single best sentence as the answer.

WHY NOT GPT/LLaMA?
  - GPT needs an API key (paid).
  - LLaMA needs 4–8 GB RAM to run locally.
  - For a college project, extractive QA is transparent, fast, and accurate
    as long as the answer exists in the document (which it always should
    in a RAG system).

CHANGES FROM ORIGINAL (rag_pipeline.py):
  - Scoring now combines: keyword overlap + position bonus + length bonus
  - Position bonus: sentences early in a chunk often contain topic sentences
  - Length bonus: very short sentences (e.g. "Yes.") are less useful
  - Returns top 2 sentences for better coverage (configurable)
  - Falls back gracefully to first chunk preview if nothing scores well
"""

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Common English stopwords for extracting key query terms.
# Intentionally small to avoid over-filtering domain queries.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has",
    "have", "having", "how", "i", "if", "in", "into", "is", "it", "its", "may",
    "might", "more", "most", "must", "no", "not", "of", "on", "or", "our", "out",
    "should", "so", "such", "than", "that", "the", "their", "then", "there",
    "these", "they", "this", "those", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "whom", "why", "will", "with", "would", "you",
    "your",
}

# Controls how many sentences to include in the answer
TOP_SENTENCES = 2

# Minimum character length for a sentence to be considered
MIN_SENTENCE_LENGTH = 15


def generate_answer(query: str, chunks: List[Dict]) -> str:
    """
    Extract the best answer sentence(s) from retrieved chunks.

    Args:
        query:  The user's original question.
        chunks: List of ranked chunk dicts (from retriever.py).
                Each has a "text" key and optionally a "score" key.

    Returns:
        A clean answer string (1–2 sentences).
    """
    if not chunks:
        return "No relevant content found. Please upload a document first."

    query_tokens = set(_tokenize(query))
    key_query_terms = _key_terms(query)

    # (score, sentence, chunk_rank, sentence_pos)
    all_scored_sentences = []

    for chunk_rank, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            continue

        sentences = _split_sentences(chunk_text)

        for sent_pos, sentence in enumerate(sentences):
            if len(sentence) < MIN_SENTENCE_LENGTH:
                continue

            # Ignore sentences that are not related to the query:
            # require at least one key query term to appear.
            if key_query_terms and not (set(_tokenize(sentence)) & key_query_terms):
                continue

            score = _score_sentence(
                sentence=sentence,
                query_tokens=query_tokens,
                position=sent_pos,
                total_sentences=len(sentences),
                chunk_rank=chunk_rank,
            )

            all_scored_sentences.append((score, sentence, chunk_rank, sent_pos))

    if not all_scored_sentences:
        logger.info("[QA] No query-related sentences found.")
        return "Answer not found in documents"

    # Sort by score descending
    all_scored_sentences.sort(key=lambda x: x[0], reverse=True)

    # Pick top N unique sentences (avoid near-duplicates)
    selected_items = []  # (sentence, chunk_rank, sentence_pos)
    selected_texts = []
    best_score = all_scored_sentences[0][0] if all_scored_sentences else 0.0
    # If the best sentence is weakly related, don't guess.
    # With our scoring (overlap + position + length - rank), strong matches are typically >= 0.55.
    if best_score < 0.55:
        logger.info(f"[QA] Best score too low ({best_score:.3f}).")
        return "Answer not found in documents"

    for score, sentence, chunk_rank, sent_pos in all_scored_sentences:
        if len(selected_items) >= TOP_SENTENCES:
            break
        if not _is_duplicate(sentence, selected_texts) and not _is_exact_duplicate(sentence, selected_texts):
            selected_items.append((sentence, chunk_rank, sent_pos))
            selected_texts.append(sentence)
            logger.debug(f"[QA] Selected (score={score:.3f}): {sentence[:80]}")

    # Re-order for readability: chunk order, then sentence order.
    if not selected_items and all_scored_sentences:
        # De-duplication can, in rare cases, filter everything out.
        best_sentence = all_scored_sentences[0][1]
        best_chunk_rank = all_scored_sentences[0][2]
        best_sent_pos = all_scored_sentences[0][3]
        selected_items = [(best_sentence, best_chunk_rank, best_sent_pos)]

    selected_items.sort(key=lambda x: (x[1], x[2]))
    answer = " ".join([s for s, _, _ in selected_items]).strip()
    logger.info(f"[QA] Final answer ({len(answer)} chars): {answer[:100]}")
    return answer


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """
    Split a chunk of text into individual sentences.
    Uses period/newline/question mark as boundaries.
    """
    # Split on '. ', '.\n', '? ', '! '
    raw = re.split(r'(?<=[.!?])\s+|\n', text)
    return [s.strip() for s in raw if s.strip()]


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens, no punctuation."""
    # Include digits to better match IDs/years/versions in queries.
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _key_terms(query: str) -> set:
    """
    Extract key query terms for strong matching.
    Drops stopwords and 1–2 character tokens to avoid noise.
    """
    tokens = _tokenize(query)
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _score_sentence(
    sentence: str,
    query_tokens: set,
    position: int,
    total_sentences: int,
    chunk_rank: int,
) -> float:
    """
    Score a sentence on three factors:

    1. Keyword overlap (0.0–1.0):
       Fraction of query words that appear in this sentence.
       Core relevance signal.

    2. Position bonus (0.0–0.2):
       Sentences near the start of a chunk often introduce the main topic.
       Slight boost for early positions.

    3. Length bonus (0.0–0.1):
       Very short sentences are usually incomplete thoughts.
       Sentences between 10–40 words are ideal.

    4. Chunk rank penalty:
       Chunks from retriever.py are already ranked best-first.
       Apply a small discount for chunks ranked lower.
    """
    sentence_tokens = set(_tokenize(sentence))

    # 1. Keyword overlap
    if not query_tokens:
        keyword_score = 0.0
    else:
        matched = query_tokens & sentence_tokens
        keyword_score = len(matched) / len(query_tokens)

    # Strongly prioritize key query terms (non-stopwords).
    key_terms = {t for t in query_tokens if len(t) >= 3 and t not in _STOPWORDS}
    if key_terms:
        key_hits = len(key_terms & sentence_tokens)
        key_ratio = key_hits / len(key_terms)
    else:
        key_ratio = 0.0

    # 2. Position bonus — earlier sentences score slightly higher
    if total_sentences > 0:
        position_bonus = 0.2 * (1 - position / total_sentences)
    else:
        position_bonus = 0.0

    # 3. Length bonus — prefer medium-length sentences (10–40 words)
    word_count = len(sentence_tokens)
    if 10 <= word_count <= 40:
        length_bonus = 0.1
    elif word_count > 40:
        length_bonus = 0.05   # long sentences ok but slightly penalized
    else:
        length_bonus = 0.0    # very short — not helpful

    # 4. Chunk rank penalty — best chunk = no penalty, 2nd = small discount, etc.
    rank_penalty = 0.05 * chunk_rank

    # Weighted blend: key term coverage dominates; position only breaks ties.
    final = (0.7 * key_ratio) + (0.3 * keyword_score) + position_bonus + length_bonus - rank_penalty
    return max(final, 0.0)


def _is_duplicate(candidate: str, selected: List[str], threshold: float = 0.6) -> bool:
    """
    Check if 'candidate' is too similar to any already-selected sentence.
    Uses word-overlap Jaccard similarity.
    Prevents the same information appearing twice in the answer.
    """
    cand_tokens = set(_tokenize(candidate))
    for existing in selected:
        existing_tokens = set(_tokenize(existing))
        union = cand_tokens | existing_tokens
        if not union:
            continue
        jaccard = len(cand_tokens & existing_tokens) / len(union)
        if jaccard >= threshold:
            return True
    return False


def _normalize_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    # Remove most punctuation for robust exact-dup detection.
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def _is_exact_duplicate(candidate: str, selected: List[str]) -> bool:
    cand_norm = _normalize_sentence(candidate)
    if not cand_norm:
        return True
    for existing in selected:
        if cand_norm == _normalize_sentence(existing):
            return True
    return False