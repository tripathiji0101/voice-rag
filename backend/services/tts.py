import asyncio
import time

from cartesia import Cartesia
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings


class SynthesisError(RuntimeError):
    """Raised when the TTS provider fails after all retries."""


class CartesiaService:
    def __init__(self):
        api_key = settings.cartesia_api_key

        if not api_key:
            raise RuntimeError("CARTESIA_API_KEY environment variable is not set")

        self.client = Cartesia(api_key=api_key)
        self.voice_id = settings.cartesia_voice_id
        self.model_id = settings.cartesia_model_id

    def _synthesize_sync(self, text: str) -> bytes:
        # .bytes() is deprecated in the installed cartesia SDK in favor of
        # .generate(), which returns a BinaryAPIResponse — .read() gives
        # the raw audio bytes.
        response = self.client.tts.generate(
            model_id=self.model_id,
            transcript=text,
            voice={"id": self.voice_id},
            output_format={
                "container": "wav",
                "encoding": "pcm_f32le",
                "sample_rate": 44100,
            },
        )
        return response.read()

    async def synthesize(self, text: str) -> tuple[bytes, float]:
        """Turn text into WAV audio bytes. Returns (audio_bytes, latency_ms)."""
        start = time.perf_counter()
        attempts = 0

        @retry(
            stop=stop_after_attempt(settings.max_retries),
            wait=wait_exponential(multiplier=settings.retry_backoff_seconds, max=4),
            reraise=True,
        )
        async def _call():
            nonlocal attempts
            attempts += 1
            return await asyncio.wait_for(
                asyncio.to_thread(self._synthesize_sync, text),
                timeout=settings.tts_timeout_seconds,
            )

        try:
            audio_bytes = await _call()
        except Exception as exc:  # noqa: BLE001
            raise SynthesisError(
                f"Text-to-speech failed after {attempts} attempt(s): {exc}"
            ) from exc

        if not audio_bytes:
            raise SynthesisError("TTS provider returned empty audio")

        return audio_bytes, (time.perf_counter() - start) * 1000
