from backend.ingestion.chunkers import chunk_fixed, chunk_recursive, chunk_sentence, get_chunker


def test_chunk_fixed_respects_overlap():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_fixed(text, chunk_size=30, overlap=10)
    assert len(chunks) > 1
    # last words of chunk N should reappear as first words of chunk N+1
    first_chunk_tail = chunks[0].split()[-10:]
    second_chunk_head = chunks[1].split()[:10]
    assert first_chunk_tail == second_chunk_head


def test_chunk_fixed_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_fixed("a b c", chunk_size=5, overlap=5)


def test_chunk_sentence_never_splits_a_sentence():
    text = (
        "FastAPI is a modern web framework. "
        "It supports async handlers natively. "
        "Pydantic validates requests automatically."
    )
    chunks = chunk_sentence(text, max_words=8, overlap_sentences=0)
    rejoined = " ".join(chunks)
    for sentence in [
        "FastAPI is a modern web framework.",
        "It supports async handlers natively.",
        "Pydantic validates requests automatically.",
    ]:
        assert sentence in rejoined


def test_chunk_recursive_keeps_short_paragraph_intact():
    text = "Paragraph one is short.\n\nParagraph two is also short."
    chunks = chunk_recursive(text, max_words=50)
    assert len(chunks) == 1
    assert "Paragraph one" in chunks[0]
    assert "Paragraph two" in chunks[0]


def test_chunk_recursive_splits_oversized_paragraph():
    long_paragraph = " ".join(f"sentence{i}." for i in range(60))
    chunks = chunk_recursive(long_paragraph, max_words=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 25 for c in chunks)  # small slack for overlap


def test_get_chunker_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError):
        get_chunker("not_a_real_strategy")


def test_empty_text_returns_no_chunks():
    assert chunk_fixed("   ") == []
    assert chunk_sentence("") == []
    assert chunk_recursive("\n\n") == []
