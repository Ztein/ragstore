"""Entrypoint: serve the ragstore API with uvicorn.

The app factory builds the service from environment configuration; the ASGI lifespan
connects SQLite + Weaviate and starts the in-process ingestion worker.
"""

from __future__ import annotations

import os

import uvicorn

from ragstore.app import create_app


def default_app():
    return create_app()


def main() -> None:
    uvicorn.run(
        "ragstore.__main__:default_app",
        factory=True,
        host=os.environ.get("RAGSTORE_HOST", "0.0.0.0"),  # noqa: S104 — container service
        port=int(os.environ.get("RAGSTORE_PORT", "8810")),
    )


if __name__ == "__main__":
    main()
