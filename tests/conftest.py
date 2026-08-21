"""Shared fixtures.

Embedding downloads (fastembed) and LLM/STT calls (Gemini, Deepgram) need
real network access and real API keys, which aren't available in every
environment (CI, sandboxes). These fixtures swap in deterministic, fully
offline fakes so the pipeline's *orchestration* -- chunking, retrieval,
guardrails, latency instrumentation, retry/fallback behavior -- can be
exercised without either.
"""
import hashlib

import numpy as np
import pytest


def _fake_vector(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-words-ish embedding: good enough that texts
    sharing vocabulary end up with higher cosine similarity than texts
    that don't, which is all the integration tests need."""
    vector = np.zeros(dim, dtype=np.float32)
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % dim] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


@pytest.fixture
def fake_embedding_service(monkeypatch):
    from backend.services import embeddings

    def fake_embed(self, texts):
        return [_fake_vector(t) for t in texts]

    monkeypatch.setattr(embeddings.EmbeddingService, "embed", fake_embed)
    monkeypatch.setattr(embeddings.EmbeddingService, "__init__", lambda self, *a, **k: None)


@pytest.fixture
def fake_llm_service(monkeypatch):
    from backend.services import llm

    async def fake_generate(self, query, context, allow_fallback=True):
        from backend.schemas import GenerationResult

        return GenerationResult(
            answer=f"[fake answer for: {query}]",
            latency_ms=1.0,
            model="fake-model",
            attempts=1,
        )

    monkeypatch.setattr(llm.LLMService, "generate", fake_generate)
    monkeypatch.setattr(llm.LLMService, "__init__", lambda self, *a, **k: setattr(self, "model", "fake-model"))


@pytest.fixture
def fake_tts_service(monkeypatch):
    from backend.services import tts

    async def fake_synthesize(self, text):
        # Minimal valid empty-ish WAV header so callers that sniff the
        # container don't choke; content doesn't need to be real audio
        # for orchestration tests.
        return b"RIFF....WAVEfmt ", 1.0

    monkeypatch.setattr(tts.CartesiaService, "synthesize", fake_synthesize)
    monkeypatch.setattr(
        tts.CartesiaService, "__init__", lambda self, *a, **k: setattr(self, "voice_id", "fake-voice")
    )


@pytest.fixture
def fake_stt_service(monkeypatch):
    from backend.services import stt
    from backend.schemas import TranscriptionResult

    async def fake_transcribe(self, audio_bytes, mimetype="audio/wav"):
        return TranscriptionResult(
            transcript="what is fastapi used for",
            confidence=0.99,
            latency_ms=1.0,
            provider="fake",
            attempts=1,
        )

    monkeypatch.setattr(stt.DeepgramService, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(stt.DeepgramService, "__init__", lambda self, *a, **k: None)
