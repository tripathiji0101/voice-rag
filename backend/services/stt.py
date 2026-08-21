import asyncio
import time

from deepgram import DeepgramClient
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.config import settings
from backend.schemas import TranscriptionResult


class TranscriptionError(RuntimeError):
    """Raised when the STT provider fails after all retries."""


class DeepgramService:
    def __init__(self):
        api_key = settings.deepgram_api_key

        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY environment variable is not set")

        self.client = DeepgramClient(api_key=api_key)

    def _transcribe_sync(self, audio_bytes: bytes, mimetype: str):
        # deepgram-sdk 7.x: ListenV1RequestFile is just an alias for
        # `bytes` (see deepgram/types/listen_v1request_file.py) and
        # transcribe_file takes it as the keyword-only `request` arg —
        # it is NOT a wrapper class you construct with buffer=...
        return self.client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model="nova-3",
            smart_format=True,
        )

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        mimetype: str = "audio/wav",
    ) -> TranscriptionResult:
        """Transcribe audio with bounded retries + a hard timeout.

        The Deepgram SDK call is synchronous/blocking, so it's run in a
        thread to avoid stalling the event loop while we wait on it.
        """
        start = time.perf_counter()
        attempts = 0

        @retry(
            stop=stop_after_attempt(settings.max_retries),
            wait=wait_exponential(multiplier=settings.retry_backoff_seconds, max=4),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        async def _call():
            nonlocal attempts
            attempts += 1
            return await asyncio.wait_for(
                asyncio.to_thread(self._transcribe_sync, audio_bytes, mimetype),
                timeout=settings.stt_timeout_seconds,
            )

        try:
            response = await _call()
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed error
            raise TranscriptionError(
                f"Speech-to-text failed after {attempts} attempt(s): {exc}"
            ) from exc

        try:
            alternative = response.results.channels[0].alternatives[0]
            transcript = alternative.transcript
            confidence = getattr(alternative, "confidence", None)
        except (AttributeError, IndexError) as exc:
            raise TranscriptionError(
                "Deepgram response did not contain a transcript"
            ) from exc

        return TranscriptionResult(
            transcript=transcript,
            confidence=confidence,
            latency_ms=(time.perf_counter() - start) * 1000,
            provider="deepgram",
            attempts=attempts,
        )
