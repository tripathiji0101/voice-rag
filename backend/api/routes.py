from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from backend.pipeline.rag import RAGPipeline
from backend.schemas import RAGResponse, VoiceRAGResponse
from backend.services.stt import DeepgramService, TranscriptionError
from backend.services.tts import CartesiaService, SynthesisError

router = APIRouter()

rag = RAGPipeline()
stt = DeepgramService()

# TTS needs its own key (CARTESIA_API_KEY) which may not be configured in
# every environment — constructed lazily so /api/query and /api/voice-query
# (text-out) still work without it.
_tts: CartesiaService | None = None


def _get_tts() -> CartesiaService:
    global _tts
    if _tts is None:
        _tts = CartesiaService()
    return _tts


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest):
    """Text-in, grounded-answer-out. Skips the STT leg entirely."""
    return await rag.run(query=request.query, top_k=request.top_k)


@router.post("/voice-query", response_model=VoiceRAGResponse)
async def voice_query(file: UploadFile = File(...), top_k: int = 5):
    """Voice-in, JSON-out: audio in, transcript + text answer back.

    Use this when the caller wants to render the answer as text (e.g. a
    chat UI showing what was heard + the response). For an audio-in,
    audio-out loop, use /voice-chat instead.
    """
    transcription, rag_result = await _run_voice_pipeline(file, top_k)

    return VoiceRAGResponse(
        transcript=transcription.transcript,
        transcription=transcription,
        **rag_result.model_dump(exclude={"query"}),
        query=rag_result.query,
    )


@router.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...), top_k: int = 5):
    """The actual "speak a question, get a spoken answer" loop.

    Audio in -> STT -> RAG -> TTS -> audio out. The response body IS the
    spoken answer (audio/wav); metadata (transcript, answer text,
    guardrail verdict, per-stage latency) rides along in response
    headers since the body is binary audio, not JSON.
    """
    transcription, rag_result = await _run_voice_pipeline(file, top_k)

    try:
        audio_bytes, tts_latency_ms = await _get_tts().synthesize(rag_result.answer)
    except SynthesisError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    total_ms = rag_result.latencies.total_ms + transcription.latency_ms + tts_latency_ms

    headers = {
        "X-Transcript": quote(transcription.transcript),
        "X-Answer": quote(rag_result.answer),
        "X-Guardrail-Verdict": rag_result.guardrail.verdict.value,
        "X-Answered": str(rag_result.answered).lower(),
        "X-Latency-Stt-Ms": f"{transcription.latency_ms:.2f}",
        "X-Latency-Retrieval-Ms": f"{rag_result.latencies.retrieval_ms:.2f}"
        if rag_result.latencies.retrieval_ms is not None
        else "",
        "X-Latency-Generation-Ms": f"{rag_result.latencies.generation_ms:.2f}"
        if rag_result.latencies.generation_ms is not None
        else "",
        "X-Latency-Tts-Ms": f"{tts_latency_ms:.2f}",
        "X-Latency-Total-Ms": f"{total_ms:.2f}",
        # Headers must be ASCII; transcript/answer are percent-encoded above
        # (browsers/clients decode with decodeURIComponent / urllib.parse.unquote).
    }

    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


async def _run_voice_pipeline(file: UploadFile, top_k: int):
    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    try:
        transcription = await stt.transcribe_audio(
            audio_bytes=audio_bytes,
            mimetype=file.content_type or "audio/wav",
        )
    except TranscriptionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    rag_result = await rag.run(query=transcription.transcript, top_k=top_k)
    return transcription, rag_result
