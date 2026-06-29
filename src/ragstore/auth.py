"""Bearer-auth ASGI middleware (hardened, mirrored from doclib).

- Accepts ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``.
- Byte-level constant-time comparison (``hmac.compare_digest``).
- Explicit allowlist of scope types: ``lifespan`` passes; ``http`` is checked;
  anything else (e.g. websocket) is denied.
- 401 is generic — no information leakage.
"""

from __future__ import annotations

import hmac
from collections.abc import Iterable

from starlette.types import ASGIApp, Receive, Scope, Send


class BearerAuthMiddleware:
    def __init__(self, app: ASGIApp, api_key: str, public_paths: Iterable[str] = ()) -> None:
        if not api_key:
            raise RuntimeError("BearerAuthMiddleware requires a non-empty api_key")
        self.app = app
        self._expected = api_key.encode("utf-8")
        self._public_paths = set(public_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope["type"]
        if scope_type == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope_type != "http":
            await self._deny_non_http(scope, send)
            return
        if scope.get("path") in self._public_paths:
            await self.app(scope, receive, send)
            return
        if self._authorized(scope):
            await self.app(scope, receive, send)
            return
        await self._unauthorized(send)

    def _authorized(self, scope: Scope) -> bool:
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        token: bytes | None = None
        auth = headers.get(b"authorization")
        if auth and auth.lower().startswith(b"bearer "):
            token = auth[7:].strip()
        elif b"x-api-key" in headers:
            token = headers[b"x-api-key"].strip()
        if not token:
            return False
        return hmac.compare_digest(token, self._expected)

    async def _unauthorized(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"detail":"Unauthorized"}'})

    async def _deny_non_http(self, scope: Scope, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
