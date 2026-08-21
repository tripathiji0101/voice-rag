"""Pre-generation guardrails: decide whether the pipeline should even
call the LLM, before spending the latency/cost on generation.

This is deliberately cheap (no network calls) so it runs inside the
sub-200ms retrieval budget and can short-circuit the expensive LLM leg
entirely when it wouldn't produce a grounded answer.

Cosine similarity alone is a noisy signal for this: general-purpose
sentence embeddings (BGE and similar) give unrelated English sentences a
non-trivial baseline similarity just from shared grammar/structure, so an
off-topic query can score *higher* than a genuinely-relevant one purely
by chance (e.g. "the capital of France" vs. "the stale smell of old
beer" against a tech corpus — the beer sentence scored higher in
practice despite having zero actual overlap with the corpus). A single
threshold on that number is therefore fragile near the boundary.

To make the block/allow decision more robust, relevance requires BOTH:
  1. cosine similarity clearing `min_relevance_score`, AND
  2. at least `min_lexical_overlap` shared, non-stopword tokens between
     the query and the top-scoring chunk.

Neither signal alone is reliable (lexical overlap misses paraphrases,
cosine similarity has the false-positive problem above), but a query
that's actually answerable from the corpus should usually satisfy both.
"""
from __future__ import annotations

import re
import time

from backend.config import settings
from backend.schemas import GuardrailResult, GuardrailVerdict, RetrievedChunk

_REFUSAL_MESSAGES = {
    GuardrailVerdict.BLOCK_EMPTY_QUERY: (
        "I didn't catch a real question there — could you rephrase it?"
    ),
    GuardrailVerdict.BLOCK_NO_CONTEXT: (
        "There's nothing in the knowledge base yet, so I can't ground an "
        "answer. Try ingesting some documents first."
    ),
    GuardrailVerdict.BLOCK_LOW_RELEVANCE: (
        "I don't have enough relevant information in the knowledge base "
        "to answer that confidently, so I'd rather not guess."
    ),
}

# Small, deliberately generic stopword list — this is a cheap filter to
# strip function words before measuring overlap, not a linguistics
# exercise. Extend if false negatives show up on real query traffic.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "do", "does", "did", "doing", "of", "in", "on", "at", "to", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "and", "or", "but", "if", "then", "so", "than", "that", "this",
    "these", "those", "it", "its", "as", "by", "from", "up", "down",
    "can", "could", "will", "would", "should", "role", "play", "best",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def refusal_message(verdict: GuardrailVerdict) -> str:
    return _REFUSAL_MESSAGES.get(
        verdict, "I don't have enough grounded information to answer that."
    )


def check(query: str, retrieved: list[RetrievedChunk]) -> GuardrailResult:
    """Run all pre-generation checks and return a single verdict.

    Checks run cheapest-first so a bad query never even touches the
    retrieval-quality check.
    """
    start = time.perf_counter()

    def finish(verdict: GuardrailVerdict, reason: str, top_score: float | None = None):
        return GuardrailResult(
            verdict=verdict,
            reason=reason,
            top_score=top_score,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    stripped = query.strip()
    if not stripped or len(stripped.split()) < settings.min_query_words:
        return finish(
            GuardrailVerdict.BLOCK_EMPTY_QUERY,
            f"Query has fewer than {settings.min_query_words} words.",
        )

    if not retrieved:
        return finish(
            GuardrailVerdict.BLOCK_NO_CONTEXT,
            "Vector store returned zero chunks (empty index or no matches).",
        )

    top_score = max(chunk.score for chunk in retrieved)
    top_chunk = max(retrieved, key=lambda chunk: chunk.score)

    cosine_pass = top_score >= settings.min_relevance_score
    supporting = sum(1 for chunk in retrieved if chunk.score >= settings.min_relevance_score)

    query_tokens = _tokenize(query)
    chunk_tokens = _tokenize(top_chunk.text)
    overlap_tokens = query_tokens & chunk_tokens
    lexical_pass = len(overlap_tokens) >= settings.min_lexical_overlap

    if not cosine_pass or supporting < settings.min_supporting_chunks:
        return finish(
            GuardrailVerdict.BLOCK_LOW_RELEVANCE,
            (
                f"Top similarity score {top_score:.3f} is below the "
                f"{settings.min_relevance_score:.2f} relevance floor; "
                "answering would likely be unsupported by the corpus."
            ),
            top_score=top_score,
        )

    if not lexical_pass:
        return finish(
            GuardrailVerdict.BLOCK_LOW_RELEVANCE,
            (
                f"Cosine score {top_score:.3f} cleared the floor but shares "
                f"no meaningful vocabulary with the top chunk (0 of "
                f"{settings.min_lexical_overlap} required overlapping "
                "terms) — likely a coincidental embedding match rather "
                "than genuine topical relevance."
            ),
            top_score=top_score,
        )

    return finish(
        GuardrailVerdict.ALLOW,
        (
            f"{supporting} chunk(s) cleared the relevance floor and shared "
            f"{len(overlap_tokens)} term(s) with the top match: "
            f"{sorted(overlap_tokens)}."
        ),
        top_score=top_score,
    )
