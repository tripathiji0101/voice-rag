"""Structured I/O contracts shared across the pipeline.

Every stage of the voice -> RAG -> answer flow speaks Pydantic, not bare
dicts, so the harness can validate inputs/outputs at each boundary and
so latency + guardrail metadata travels with the result instead of
being bolted on afterward.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"


class Chunk(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# STT
# --------------------------------------------------------------------------
class TranscriptionResult(BaseModel):
    transcript: str
    confidence: Optional[float] = None
    duration_seconds: Optional[float] = None
    latency_ms: float
    provider: str = "deepgram"
    attempts: int = 1


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float


class RetrievalResult(BaseModel):
    results: list[RetrievedChunk]
    latency_ms: float
    top_k: int
    query_embed_latency_ms: float
    search_latency_ms: float


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------
class GuardrailVerdict(str, Enum):
    ALLOW = "allow"
    BLOCK_LOW_RELEVANCE = "block_low_relevance"
    BLOCK_EMPTY_QUERY = "block_empty_query"
    BLOCK_NO_CONTEXT = "block_no_context"
    BLOCK_UNSAFE = "block_unsafe"


class GuardrailResult(BaseModel):
    verdict: GuardrailVerdict
    reason: str
    top_score: Optional[float] = None
    latency_ms: float


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
class GenerationResult(BaseModel):
    answer: str
    latency_ms: float
    model: str
    attempts: int = 1
    fallback_used: bool = False


# --------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------
class StageLatencies(BaseModel):
    stt_ms: Optional[float] = None
    query_embed_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    guardrail_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    tts_ms: Optional[float] = None
    total_ms: float


class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: list[RetrievedChunk]
    guardrail: GuardrailResult
    latencies: StageLatencies
    answered: bool


class VoiceRAGResponse(RAGResponse):
    transcript: str
    transcription: TranscriptionResult
