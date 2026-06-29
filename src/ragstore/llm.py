"""Generation LLM client — the only place an LLM enters the component.

OpenAI-compatible chat completions over httpx. Provider-agnostic via
LLM_BASE_URL / LLM_MODEL / LLM_API_KEY. Fail loud on any non-200.
"""

from __future__ import annotations

from typing import Any

import httpx

from ragstore.config import Settings, require_llm


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 120.0) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMClient:
        base_url, model, api_key = require_llm(settings)
        return cls(base_url=base_url, model=model, api_key=api_key)

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {"model": model or self._model, "messages": messages}
        if params:
            body.update(params)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url, json=body, headers={"Authorization": f"Bearer {self._api_key}"}
            )
        if resp.status_code != 200:
            raise RuntimeError(f"LLM provider returned {resp.status_code}: {resp.text[:500]}")
        return resp.json()["choices"][0]["message"]["content"]
