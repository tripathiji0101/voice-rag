"""Multiple chunking strategies for retrieval-quality experimentation.

Naive fixed-size word splitting ignores document structure and regularly
cuts sentences (and facts) in half. This module gives the ingestion
pipeline a choice of strategies, each suited to different content:

- ``fixed``:     fixed-size sliding window over words. Cheap, predictable,
                 good baseline / fallback for unstructured text.
- ``sentence``:  packs whole sentences into a chunk until the target size
                 is hit, so a fact never gets split mid-sentence.
- ``recursive``: LangChain-style recursive splitting. Tries to split on
                 paragraph boundaries first, falls back to sentences,
                 then words, only crossing a bigger boundary when a
                 smaller one can't fit the content in-budget. Preserves
                 as much semantic structure as possible.
"""
from __future__ import annotations

import re
from typing import Callable

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Za-z0-9\"'])")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = _PARAGRAPH_BOUNDARY.split(text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_fixed(
    text: str,
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[str]:
    """Fixed-size sliding window over whitespace-split words."""
    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_sentence(
    text: str,
    max_words: int = 180,
    overlap_sentences: int = 1,
) -> list[str]:
    """Pack whole sentences together until ``max_words`` is reached.

    Never splits a sentence in half. Carries the last ``overlap_sentences``
    sentences of a chunk into the next chunk so retrieval doesn't lose
    context at a boundary.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(s.split()) for s in current)

        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_recursive(
    text: str,
    max_words: int = 200,
    overlap_sentences: int = 1,
) -> list[str]:
    """Split on the largest structural boundary that still fits the budget.

    Tries paragraphs first (keeps a whole idea together). Any paragraph
    that's still too big gets recursed into sentence-level packing via
    :func:`chunk_sentence`. This keeps chunks close to ``max_words`` while
    disturbing document structure as little as possible.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return chunk_sentence(text, max_words=max_words, overlap_sentences=overlap_sentences)

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0

    def flush_buffer():
        if buffer:
            chunks.append("\n\n".join(buffer))

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.split())

        if paragraph_words > max_words:
            # Paragraph alone busts the budget: flush what we have, then
            # recurse into sentence-level packing for this paragraph.
            flush_buffer()
            buffer = []
            buffer_words = 0
            chunks.extend(
                chunk_sentence(
                    paragraph,
                    max_words=max_words,
                    overlap_sentences=overlap_sentences,
                )
            )
            continue

        if buffer_words + paragraph_words > max_words:
            flush_buffer()
            buffer = [paragraph]
            buffer_words = paragraph_words
        else:
            buffer.append(paragraph)
            buffer_words += paragraph_words

    flush_buffer()
    return chunks


# Backwards-compatible name used by earlier code / callers.
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    return chunk_fixed(text, chunk_size=chunk_size, overlap=overlap)


CHUNKERS: dict[str, Callable[..., list[str]]] = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "recursive": chunk_recursive,
}


def get_chunker(strategy: str) -> Callable[..., list[str]]:
    try:
        return CHUNKERS[strategy]
    except KeyError as exc:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Available: {sorted(CHUNKERS)}"
        ) from exc
