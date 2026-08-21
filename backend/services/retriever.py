import time

from backend.services.embeddings import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.schemas import RetrievalResult, RetrievedChunk


class Retriever:
    def __init__(
        self,
        vector_store_path: str = "vector_data/store.json",
    ):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(vector_store_path)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        start = time.perf_counter()

        if not query.strip():
            now = time.perf_counter()
            return RetrievalResult(
                results=[],
                latency_ms=(now - start) * 1000,
                top_k=top_k,
                query_embed_latency_ms=0.0,
                search_latency_ms=0.0,
            )

        embed_start = time.perf_counter()
        query_embedding = self.embedding_service.embed([query])[0]
        embed_latency_ms = (time.perf_counter() - embed_start) * 1000

        search_start = time.perf_counter()
        raw_results = self.vector_store.search(query_embedding, top_k=top_k)
        search_latency_ms = (time.perf_counter() - search_start) * 1000

        return RetrievalResult(
            results=[RetrievedChunk(**result) for result in raw_results],
            latency_ms=(time.perf_counter() - start) * 1000,
            top_k=top_k,
            query_embed_latency_ms=embed_latency_ms,
            search_latency_ms=search_latency_ms,
        )
