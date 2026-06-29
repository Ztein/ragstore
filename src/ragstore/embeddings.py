"""Embedding client — component-owned, external OpenAI-compatible provider.

Fail loud: any non-200 from the provider, or a returned vector whose dimension does
not match ``EMBEDDING_DIM``, raises. No fallback to a local model.
"""

from __future__ import annotations

import httpx

from ragstore.config import Settings


class EmbeddingClient:
    def __init__(
        self, base_url: str, model: str, api_key: str, dim: int, timeout: float = 60.0
    ) -> None:
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._api_key = api_key
        self._dim = dim
        self._timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings) -> EmbeddingClient:
        return cls(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            dim=settings.embedding_dim,
        )

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url,
                json={"model": self._model, "input": texts},
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        if resp.status_code != 200:
            raise RuntimeError(f"embedding provider returned {resp.status_code}: {resp.text[:500]}")
        data = resp.json()["data"]
        vectors = [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"embedding provider returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        for v in vectors:
            if len(v) != self._dim:
                raise RuntimeError(
                    f"embedding dimension mismatch: expected {self._dim}, got {len(v)}"
                )
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
