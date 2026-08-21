"""End-to-end orchestration tests using offline fakes for the embedding
model and the LLM (see conftest.py). These prove the harness itself --
retrieval -> guardrail -> generation -> structured response with per-stage
latency -- works, independent of any third-party network availability.
"""
import pytest

from backend.ingestion.pipeline import ingest_documents
from backend.pipeline.rag import RAGPipeline
from backend.schemas import GuardrailVerdict


@pytest.fixture
def ingested_store(tmp_path, fake_embedding_service):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "doc.txt").write_text(
        "FastAPI is a modern Python web framework for building APIs. "
        "It supports asynchronous request handlers natively.",
        encoding="utf-8",
    )
    store_path = tmp_path / "store.json"
    count = ingest_documents(
        directory=str(data_dir),
        vector_store_path=str(store_path),
        strategy="sentence",
    )
    assert count > 0
    return str(store_path)


@pytest.mark.asyncio
async def test_pipeline_answers_relevant_query(
    ingested_store, fake_embedding_service, fake_llm_service, monkeypatch
):
    from backend.config import settings
    from backend.services.vector_store import VectorStore

    # The fake hashed bag-of-words embedding is a much cruder similarity
    # signal than the real sentence embedding model; relax the floor so
    # this test exercises the ALLOW path deterministically.
    monkeypatch.setattr(settings, "min_relevance_score", 0.05)

    pipeline = RAGPipeline()
    pipeline.retriever.vector_store = VectorStore(ingested_store)

    result = await pipeline.run("What is FastAPI used for?", top_k=3)

    assert result.answered is True
    assert result.guardrail.verdict == GuardrailVerdict.ALLOW
    assert result.latencies.total_ms >= 0
    assert result.latencies.generation_ms is not None
    assert "fake answer" in result.answer


@pytest.mark.asyncio
async def test_pipeline_refuses_short_query(ingested_store, fake_embedding_service, fake_llm_service):
    pipeline = RAGPipeline()
    from backend.services.vector_store import VectorStore

    pipeline.retriever.vector_store = VectorStore(ingested_store)

    result = await pipeline.run("hi", top_k=3)

    assert result.answered is False
    assert result.guardrail.verdict == GuardrailVerdict.BLOCK_EMPTY_QUERY
    # Guardrail should short-circuit before the (network-bound) LLM leg.
    assert result.latencies.generation_ms is None


@pytest.mark.asyncio
async def test_pipeline_refuses_empty_index(tmp_path, fake_embedding_service, fake_llm_service):
    from backend.services.vector_store import VectorStore

    pipeline = RAGPipeline()
    pipeline.retriever.vector_store = VectorStore(str(tmp_path / "empty.json"))

    result = await pipeline.run("What is FastAPI used for?", top_k=3)

    assert result.answered is False
    assert result.guardrail.verdict == GuardrailVerdict.BLOCK_NO_CONTEXT
