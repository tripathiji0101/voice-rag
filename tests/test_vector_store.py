import numpy as np

from backend.services.vector_store import VectorStore


def make_store(tmp_path):
    return VectorStore(str(tmp_path / "store.json"))


def test_add_and_search_returns_best_match(tmp_path):
    store = make_store(tmp_path)
    store.add(
        texts=["cats are mammals", "the stock market fell today"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        metadatas=[{"id": 1}, {"id": 2}],
    )

    results = store.search([0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "cats are mammals"
    assert results[0]["score"] > 0.9


def test_search_on_empty_store_returns_empty(tmp_path):
    store = make_store(tmp_path)
    assert store.search([1.0, 0.0], top_k=5) == []


def test_search_respects_top_k(tmp_path):
    store = make_store(tmp_path)
    n = 10
    rng = np.random.default_rng(0)
    embeddings = rng.random((n, 8)).tolist()
    store.add(
        texts=[f"doc{i}" for i in range(n)],
        embeddings=embeddings,
        metadatas=[{} for _ in range(n)],
    )
    results = store.search(embeddings[0], top_k=3)
    assert len(results) == 3


def test_persists_and_reloads(tmp_path):
    path = tmp_path / "store.json"
    store = VectorStore(str(path))
    store.add(texts=["a"], embeddings=[[1.0, 0.0]], metadatas=[{"k": "v"}])

    reloaded = VectorStore(str(path))
    assert len(reloaded) == 1
    assert reloaded.documents[0]["text"] == "a"
