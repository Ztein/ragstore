"""Embedding client against a real (fake) OpenAI-compatible provider over HTTP."""

import httpx
import pytest

from ragstore.embeddings import EmbeddingClient

DIM = 8


def _client(fake_provider, dim=DIM):
    return EmbeddingClient(base_url=fake_provider.base_url, model="m", api_key="k", dim=dim)


async def test_embed_returns_vectors_of_expected_dim(fake_provider):
    client = _client(fake_provider)
    vectors = await client.embed(["apple banana", "carrot potato"])
    assert len(vectors) == 2
    assert all(len(v) == DIM for v in vectors)


async def test_embed_is_deterministic(fake_provider):
    client = _client(fake_provider)
    a = await client.embed_one("same text")
    b = await client.embed_one("same text")
    assert a == b


async def test_embed_empty_returns_empty(fake_provider):
    client = _client(fake_provider)
    assert await client.embed([]) == []


async def test_dim_mismatch_fails_loud(fake_provider):
    client = _client(fake_provider, dim=999)
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        await client.embed_one("hello")


async def test_provider_error_fails_loud():
    client = EmbeddingClient(
        base_url="http://127.0.0.1:1/v1", model="m", api_key="k", dim=DIM, timeout=1.0
    )
    with pytest.raises(httpx.RequestError):  # connection error propagates, not swallowed
        await client.embed_one("hello")
