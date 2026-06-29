"""Embedding client against the real external provider over HTTP — no fakes."""

import httpx
import pytest

from ragstore.embeddings import EmbeddingClient


async def test_embed_returns_vectors_of_expected_dim(embedder):
    vectors = await embedder.embed(["apple banana", "carrot potato"])
    assert len(vectors) == 2
    assert all(len(v) == embedder.dim for v in vectors)


async def test_embed_is_deterministic(embedder):
    a = await embedder.embed_one("same text")
    b = await embedder.embed_one("same text")
    assert a == b


async def test_embed_empty_returns_empty(embedder):
    assert await embedder.embed([]) == []


async def test_dim_mismatch_fails_loud(provider):
    client = EmbeddingClient(
        base_url=provider["embedding_base_url"],
        model=provider["embedding_model"],
        api_key=provider["embedding_api_key"],
        dim=provider["embedding_dim"] + 1,  # deliberately wrong
    )
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        await client.embed_one("hello")


async def test_provider_error_fails_loud(provider):
    client = EmbeddingClient(
        base_url="http://127.0.0.1:1/v1",
        model=provider["embedding_model"],
        api_key="k",
        dim=provider["embedding_dim"],
        timeout=1.0,
    )
    with pytest.raises(httpx.RequestError):  # connection error propagates, not swallowed
        await client.embed_one("hello")
