import json
from pathlib import Path

import numpy as np


class VectorStore:
    """JSON-backed vector store with a matrix-multiply search path.

    Small enough for a demo corpus to stay entirely in memory; the search
    matters most for the latency budget so it's done as one vectorized
    matmul instead of a per-document Python loop.
    """

    def __init__(self, storage_path: str = "vector_data/store.json"):
        self.storage_path = Path(storage_path)
        self.documents: list[dict] = []
        self._matrix: np.ndarray | None = None
        self.load()

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length")

        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            self.documents.append(
                {
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata,
                }
            )

        self._matrix = None  # invalidate cache
        self.save()

    def _build_matrix(self) -> np.ndarray | None:
        if not self.documents:
            return None

        matrix = np.array(
            [document["embedding"] for document in self.documents],
            dtype=np.float32,
        )
        # Normalize once so search is a plain dot product.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return matrix / norms

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        if not self.documents:
            return []

        if self._matrix is None:
            self._matrix = self._build_matrix()

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        scores = self._matrix @ query  # cosine similarity, vectorized
        top_k = min(top_k, len(self.documents))
        # argpartition avoids a full sort for large corpora.
        candidate_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        candidate_idx = candidate_idx[np.argsort(-scores[candidate_idx])]

        return [
            {
                "text": self.documents[i]["text"],
                "metadata": self.documents[i]["metadata"],
                "score": float(scores[i]),
            }
            for i in candidate_idx
        ]

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.storage_path, "w", encoding="utf-8") as file:
            json.dump(self.documents, file, ensure_ascii=False, indent=2)

    def load(self) -> None:
        self._matrix = None

        if not self.storage_path.exists():
            self.documents = []
            return

        with open(self.storage_path, "r", encoding="utf-8") as file:
            self.documents = json.load(file)

    def __len__(self) -> int:
        return len(self.documents)
