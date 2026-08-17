from fastembed import TextEmbedding


class EmbeddingService:
    def __init__(self):
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.embed(texts)
        return [embedding.tolist() for embedding in embeddings]
