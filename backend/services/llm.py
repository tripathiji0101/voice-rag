import asyncio
import time

from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings
from backend.schemas import GenerationResult

_PROMPT_TEMPLATE = """You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Use the context as the sole source of information.
- Do not invent facts that are not supported by the context.
- If the context does not contain enough information, say so plainly
  instead of guessing.
- Give a clear and concise answer.

Context:
{context}

User question:
{query}
"""

_FALLBACK_ANSWER = (
    "I retrieved relevant context but couldn't reach the language model "
    "to compose an answer right now. Here is the raw supporting context "
    "instead:\n\n{context}"
)


class GenerationError(RuntimeError):
    """Raised when the LLM provider fails after all retries."""


class LLMService:
    def __init__(self):
        api_key = settings.gemini_api_key

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")

        self.client = genai.Client(api_key=api_key)
        self.model = settings.gemini_model

    def _generate_sync(self, prompt: str):
        return self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

    async def generate(
        self,
        query: str,
        context: str,
        allow_fallback: bool = True,
    ) -> GenerationResult:
        start = time.perf_counter()
        attempts = 0
        prompt = _PROMPT_TEMPLATE.format(context=context, query=query)

        @retry(
            stop=stop_after_attempt(settings.max_retries),
            wait=wait_exponential(multiplier=settings.retry_backoff_seconds, max=4),
            reraise=True,
        )
        async def _call():
            nonlocal attempts
            attempts += 1
            return await asyncio.wait_for(
                asyncio.to_thread(self._generate_sync, prompt),
                timeout=settings.llm_timeout_seconds,
            )

        try:
            response = await _call()
            answer = response.text
            if not answer:
                raise GenerationError("LLM returned an empty response")

            return GenerationResult(
                answer=answer,
                latency_ms=(time.perf_counter() - start) * 1000,
                model=self.model,
                attempts=attempts,
            )
        except Exception as exc:  # noqa: BLE001
            if not allow_fallback:
                raise GenerationError(
                    f"Generation failed after {attempts} attempt(s): {exc}"
                ) from exc

            # Degrade gracefully: surface the retrieved context directly
            # rather than a hard 500, so the caller still gets something
            # grounded even when the LLM leg is down.
            return GenerationResult(
                answer=_FALLBACK_ANSWER.format(context=context),
                latency_ms=(time.perf_counter() - start) * 1000,
                model=self.model,
                attempts=attempts,
                fallback_used=True,
            )
