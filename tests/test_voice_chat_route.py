"""Smoke test for the actual speak-in/listen-out loop (/api/voice-chat).

Uses offline fakes for STT, embeddings, and TTS (see conftest.py) so this
runs without real API keys or network access, while still exercising the
real route wiring: audio upload -> transcript -> RAG -> synthesized audio
-> response with metadata in headers.
"""
import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(
    tmp_path,
    fake_embedding_service,
    fake_llm_service,
    fake_tts_service,
    fake_stt_service,
    monkeypatch,
):
    from backend.config import settings
    from backend.ingestion.pipeline import ingest_documents
    from backend.services.vector_store import VectorStore

    monkeypatch.setattr(settings, "min_relevance_score", 0.02)
    monkeypatch.setattr(settings, "min_lexical_overlap", 0)

    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "doc.txt").write_text(
        "FastAPI is a modern Python web framework for building APIs.",
        encoding="utf-8",
    )
    store_path = tmp_path / "store.json"
    ingest_documents(directory=str(data_dir), vector_store_path=str(store_path), strategy="sentence")

    import backend.api.routes as routes_module
    from backend.main import app

    routes_module.rag.retriever.vector_store = VectorStore(str(store_path))

    return TestClient(app)


def test_voice_chat_returns_audio_with_metadata_headers(client):
    fake_audio = io.BytesIO(b"not real audio bytes, just needs to be non-empty")
    response = client.post(
        "/api/voice-chat",
        files={"file": ("question.wav", fake_audio, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 0

    from urllib.parse import unquote

    assert unquote(response.headers["x-transcript"]) == "what is fastapi used for"
    assert "fake answer" in unquote(response.headers["x-answer"])
    assert response.headers["x-answered"] == "true"
    assert float(response.headers["x-latency-total-ms"]) >= 0
