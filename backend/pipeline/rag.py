import time

from backend import guardrails
from backend.config import settings
from backend.schemas import (
    GuardrailVerdict,
    RAGResponse,
    StageLatencies,
)
from backend.services.llm import LLMService
from backend.services.retriever import Retriever


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLMService()

    def build_context(self, results) -> str:
        if not results:
            return ""

        return "\n\n".join(
            f"[Source {index}]\n{result.text}"
            for index, result in enumerate(results, start=1)
        )

    async def run(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RAGResponse:
        total_start = time.perf_counter()
        top_k = top_k or settings.default_top_k

        # 1. Retrieve (embed query + vector search) — local, no network.
        retrieval = self.retriever.retrieve(query=query, top_k=top_k)

        # 2. Guardrail check BEFORE spending latency/cost on generation.
        guardrail_result = guardrails.check(query, retrieval.results)

        if guardrail_result.verdict != GuardrailVerdict.ALLOW:
            total_ms = (time.perf_counter() - total_start) * 1000
            return RAGResponse(
                query=query,
                answer=guardrails.refusal_message(guardrail_result.verdict),
                sources=retrieval.results,
                guardrail=guardrail_result,
                latencies=StageLatencies(
                    query_embed_ms=retrieval.query_embed_latency_ms,
                    retrieval_ms=retrieval.latency_ms,
                    guardrail_ms=guardrail_result.latency_ms,
                    generation_ms=None,
                    total_ms=total_ms,
                ),
                answered=False,
            )

        # 3. Build context + generate (network-bound LLM call).
        context = self.build_context(retrieval.results)
        generation = await self.llm.generate(query=query, context=context)

        total_ms = (time.perf_counter() - total_start) * 1000

        return RAGResponse(
            query=query,
            answer=generation.answer,
            sources=retrieval.results,
            guardrail=guardrail_result,
            latencies=StageLatencies(
                query_embed_ms=retrieval.query_embed_latency_ms,
                retrieval_ms=retrieval.latency_ms,
                guardrail_ms=guardrail_result.latency_ms,
                generation_ms=generation.latency_ms,
                total_ms=total_ms,
            ),
            answered=True,
        )
