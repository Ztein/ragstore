"""Word-based chunking with configurable size and overlap.

Mirrors doclib's simple, predictable chunker. Returns chunks in order; each chunk is
``chunk_size`` words with ``chunk_overlap`` words shared with the previous chunk.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks
