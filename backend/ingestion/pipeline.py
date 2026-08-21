from pathlib import Path

from backend.ingestion.chunkers import get_chunker
from backend.ingestion.loaders import load_documents
from backend.services.embeddings import EmbeddingService
from backend.services.vector_store import VectorStore


def ingest_documents(
    directory: str | Path = "data/raw",
    vector_store_path: str = "vector_data/store.json",
    strategy: str = "recursive",
    **chunker_kwargs,
) -> int:
    """Load, chunk, embed and persist every .txt document in ``directory``.

    ``strategy`` selects one of the chunking strategies registered in
    ``backend.ingestion.chunkers`` ("fixed", "sentence", "recursive").
    Each stored chunk is tagged with the strategy that produced it so
    retrieval results can be inspected/compared later.
    """
    documents = load_documents(directory)

    if not documents:
        return 0

    chunker = get_chunker(strategy)

    all_chunks = []
    all_metadata = []

    for document in documents:
        chunks = chunker(document["text"], **chunker_kwargs)

        for index, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            all_metadata.append(
                {
                    **document["metadata"],
                    "chunk_index": index,
                    "chunk_strategy": strategy,
                }
            )

    if not all_chunks:
        return 0

    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed(all_chunks)

    store_path = Path(vector_store_path)

    if store_path.exists():
        store_path.unlink()

    store = VectorStore(vector_store_path)

    store.add(
        texts=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata,
    )

    return len(all_chunks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument("--directory", default="data/raw")
    parser.add_argument("--vector-store-path", default="vector_data/store.json")
    parser.add_argument(
        "--strategy", default="recursive", choices=["fixed", "sentence", "recursive"]
    )
    args = parser.parse_args()

    count = ingest_documents(
        directory=args.directory,
        vector_store_path=args.vector_store_path,
        strategy=args.strategy,
    )
    print(f"Ingested {count} chunks using '{args.strategy}' strategy -> {args.vector_store_path}")
