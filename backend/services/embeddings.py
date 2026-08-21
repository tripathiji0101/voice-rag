from fastembed import TextEmbedding


class EmbeddingService:
    """Thin wrapper around fastembed, kept process-wide singleton-friendly.

    Loading the ONNX model is the expensive part (~seconds); callers should
    construct this once (see ``Retriever``) rather than per-request.
    """

    _shared_model: TextEmbedding | None = None
    _shared_model_name: str | None = None

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        # Reuse a process-wide model instance across EmbeddingService
        # instances so re-instantiating the retriever doesn't reload
        # weights on every request.
        if (
            EmbeddingService._shared_model is None
            or EmbeddingService._shared_model_name != model_name
        ):
            EmbeddingService._shared_model = TextEmbedding(model_name=model_name)
            EmbeddingService._shared_model_name = model_name

        self.model = EmbeddingService._shared_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.embed(texts)
        return [embedding.tolist() for embedding in embeddings]
