from backend import guardrails
from backend.config import settings
from backend.schemas import GuardrailVerdict, RetrievedChunk


def test_blocks_empty_query():
    result = guardrails.check("hi", [])
    assert result.verdict == GuardrailVerdict.BLOCK_EMPTY_QUERY


def test_blocks_when_no_chunks_retrieved():
    result = guardrails.check("what is fastapi used for", [])
    assert result.verdict == GuardrailVerdict.BLOCK_NO_CONTEXT


def test_blocks_low_cosine_score():
    chunks = [RetrievedChunk(text="FastAPI is a web framework", metadata={}, score=0.1)]
    result = guardrails.check("what is fastapi used for", chunks)
    assert result.verdict == GuardrailVerdict.BLOCK_LOW_RELEVANCE


def test_allows_high_relevance_with_real_overlap():
    chunks = [RetrievedChunk(text="FastAPI is a web framework for building APIs", metadata={}, score=0.8)]
    result = guardrails.check("what is fastapi used for", chunks)
    assert result.verdict == GuardrailVerdict.ALLOW


def test_blocks_high_cosine_but_zero_lexical_overlap(monkeypatch):
    """Regression test: an unrelated sentence can score ABOVE the cosine
    floor purely from shared grammatical structure (observed with
    "the stale smell of old beer lingers" scoring higher than
    "capital of France" against a tech corpus). The lexical-overlap
    check should catch what cosine alone misses."""
    monkeypatch.setattr(settings, "min_relevance_score", 0.3)
    chunks = [
        RetrievedChunk(
            text="Retrieval-augmented generation combines a retrieval step with a language model",
            metadata={},
            score=0.41,  # clears the (lowered) cosine floor
        )
    ]
    result = guardrails.check("the stale smell of old beer lingers", chunks)
    assert result.verdict == GuardrailVerdict.BLOCK_LOW_RELEVANCE
    assert "lexical" in result.reason.lower() or "vocabulary" in result.reason.lower()


def test_refusal_message_is_non_empty_for_every_block_verdict():
    for verdict in GuardrailVerdict:
        if verdict == GuardrailVerdict.ALLOW:
            continue
        assert guardrails.refusal_message(verdict)
