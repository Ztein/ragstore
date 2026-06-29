"""A tiny, real OpenAI-compatible HTTP server used as a stand-in provider in tests.

This is a *test fixture*, not product code — it lets tests run truly end-to-end over
real HTTP without baking any fake/offline mode into ragstore itself. Embeddings are
deterministic per text (a hashing bag-of-words) so the same text retrieves the same
document; overlapping words produce closer vectors.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time

import uvicorn
from fastapi import FastAPI, Request


def _embed(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def build_app(dim: int = 8) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/embeddings")
    async def embeddings(req: Request):
        body = await req.json()
        inputs = body["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        data = [{"object": "embedding", "index": i, "embedding": _embed(t, dim)}
                for i, t in enumerate(inputs)]
        return {"object": "list", "data": data, "model": body.get("model")}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        last = body["messages"][-1]["content"]
        answer = f"ANSWER[{last[:80]}]"
        return {
            "id": "fake",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
        }

    return app


class FakeProvider:
    """Runs the fake provider on a real port in a background thread."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""

    def start(self) -> str:
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        config = uvicorn.Config(build_app(self.dim), host="127.0.0.1", port=port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        while not self._server.started:
            time.sleep(0.02)
        self.base_url = f"http://127.0.0.1:{port}/v1"
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
