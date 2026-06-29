"""Shared test fixtures.

E2E-first: tests run against a real SQLite file and (where marked) a real Weaviate
instance. No fakes baked into product code — the only "fake" is a local HTTP server
standing in for the embedding/LLM provider, used by provider tests.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fake_provider import FakeProvider

from ragstore.sqlite_store import SqliteStore
from ragstore.weaviate_store import WeaviateStore

WEAVIATE = dict(http_host="localhost", http_port=8080, grpc_host="localhost", grpc_port=50051)
EMBEDDING_DIM = 8


@pytest.fixture(scope="session")
def fake_provider():
    provider = FakeProvider(dim=EMBEDDING_DIM)
    provider.start()
    yield provider
    provider.stop()


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SqliteStore(str(tmp_path / "ragstore.db"))
    await s.connect()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def service(store, wstore, fake_provider):
    from ragstore.config import load_settings
    from ragstore.embeddings import EmbeddingClient
    from ragstore.service import RagService

    settings = load_settings(
        _env_file=None,
        ragstore_api_key="test-key",
        sqlite_path="unused",
        embedding_base_url=fake_provider.base_url,
        embedding_model="m",
        embedding_api_key="k",
        embedding_dim=EMBEDDING_DIM,
        llm_base_url=fake_provider.base_url,
        llm_model="m",
        llm_api_key="k",
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
