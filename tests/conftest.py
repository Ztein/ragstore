"""Shared test fixtures.

No mocks, no fakes — tests run against the **real** dependencies: a real Weaviate
instance, a real SQLite file, and the **real** external embedding/LLM provider
(OpenAI-compatible). Provider config comes from the environment / a local `.env`.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from ragstore.sqlite_store import SqliteStore
from ragstore.weaviate_store import WeaviateStore

# Load local .env so real provider credentials are available to the tests.
load_dotenv()

WEAVIATE = dict(http_host="localhost", http_port=8080, grpc_host="localhost", grpc_port=50051)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set — required to test against the real provider")
    return value


@pytest.fixture(scope="session")
def provider() -> dict:
    """Real embedding + LLM provider config (OpenAI-compatible). No fake fallback."""
    return {
        "embedding_base_url": _require_env("EMBEDDING_BASE_URL"),
        "embedding_model": _require_env("EMBEDDING_MODEL"),
        "embedding_api_key": _require_env("EMBEDDING_API_KEY"),
        "embedding_dim": int(_require_env("EMBEDDING_DIM")),
        "llm_base_url": _require_env("LLM_BASE_URL"),
        "llm_model": _require_env("LLM_MODEL"),
        "llm_api_key": _require_env("LLM_API_KEY"),
    }


@pytest.fixture(scope="session")
def embedder(provider):
    from ragstore.embeddings import EmbeddingClient

    return EmbeddingClient(
        base_url=provider["embedding_base_url"],
        model=provider["embedding_model"],
        api_key=provider["embedding_api_key"],
        dim=provider["embedding_dim"],
    )


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SqliteStore(str(tmp_path / "ragstore.db"))
    await s.connect()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def wstore():
    s = WeaviateStore(**WEAVIATE)
    try:
        await s.connect()
    except Exception as exc:  # noqa: BLE001 — infra gate, not product code
        host, port = WEAVIATE["http_host"], WEAVIATE["http_port"]
        pytest.skip(f"Weaviate not reachable at {host}:{port}: {exc}")
    yield s
    await s.close()


@pytest_asyncio.fixture
async def service(store, wstore, provider):
    from ragstore.config import load_settings
    from ragstore.embeddings import EmbeddingClient
    from ragstore.service import RagService

    settings = load_settings(
        _env_file=None,
        ragstore_api_key="test-key",
        sqlite_path="unused",
        **provider,
    )
    emb = EmbeddingClient.from_settings(settings)
    return RagService(settings, store, wstore, emb)


@pytest_asyncio.fixture
async def worker(service):
    from ragstore.worker import IngestionWorker

    return IngestionWorker(service)


@pytest_asyncio.fixture
async def api(service):
    from httpx import ASGITransport, AsyncClient

    from ragstore.app import create_app

    app = create_app(service)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": "Bearer test-key"},
    ) as client:
        yield client
