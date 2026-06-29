"""FastAPI application: the HTTP surface mirroring the PRD interface.

Wiring: a lifespan builds the service (connect SQLite + Weaviate) and starts the
in-process ingestion worker. In tests a pre-built service is injected. Bearer auth
guards everything except /health and /ready.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ragstore.auth import BearerAuthMiddleware
from ragstore.config import Settings, get_settings
from ragstore.embeddings import EmbeddingClient
from ragstore.service import NotFoundError, RagService
from ragstore.sqlite_store import SqliteStore
from ragstore.weaviate_store import WeaviateStore

PUBLIC_PATHS = {"/health", "/ready"}


# ---- request models ------------------------------------------------------


class CreateCollectionRequest(BaseModel):
    name: str
    config: dict[str, Any] | None = None


class UpsertRequest(BaseModel):
    external_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    version_token: str | None = None


class DeleteByFilterRequest(BaseModel):
    filter: dict[str, Any]


class QueryRequest(BaseModel):
    collection_ids: list[str]
    text: str
    k: int = 5
    filters: dict[str, Any] | None = None
    min_score: float | None = None
    mode: str = "hybrid"
    rerank: bool = False
    principal: str | None = None
    generate: dict[str, Any] | None = None


# ---- service wiring -------------------------------------------------------


async def build_service_from_env(settings: Settings) -> RagService:
    sqlite = SqliteStore(settings.sqlite_path)
    await sqlite.connect()
    weaviate = WeaviateStore(
        http_host=settings.weaviate_http_host,
        http_port=settings.weaviate_http_port,
        http_secure=settings.weaviate_http_secure,
        grpc_host=settings.weaviate_grpc_host,
        grpc_port=settings.weaviate_grpc_port,
        grpc_secure=settings.weaviate_grpc_secure,
        api_key=settings.weaviate_api_key,
    )
    await weaviate.connect()
    embeddings = EmbeddingClient.from_settings(settings)
    return RagService(settings, sqlite, weaviate, embeddings)


def get_service(request: Request) -> RagService:
    return request.app.state.service


def create_app(service: RagService | None = None) -> FastAPI:
    settings = service.settings if service is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from ragstore.worker import IngestionWorker

        owns_service = service is None
        svc = service if service is not None else await build_service_from_env(settings)
        app.state.service = svc
        worker = IngestionWorker(svc)
        worker.start()
        app.state.worker = worker
        try:
            yield
        finally:
            await worker.stop()
            if owns_service:
                await svc.aclose()

    app = FastAPI(title="ragstore", version="0.1.0", lifespan=lifespan)
    if service is not None:
        # Make the injected service usable without running the ASGI lifespan
        # (tests drive the worker explicitly).
        app.state.service = service
    app.add_middleware(
        BearerAuthMiddleware, api_key=settings.ragstore_api_key, public_paths=PUBLIC_PATHS
    )

    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _bad_request(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    # ---- collections -----------------------------------------------------

    @app.post("/collections", status_code=201)
    async def create_collection(
        body: CreateCollectionRequest, svc: RagService = Depends(get_service)
    ):
        cid = await svc.create_collection(body.name, body.config)
        return {"id": cid}

    @app.get("/collections")
    async def list_collections(svc: RagService = Depends(get_service)):
        return [
            {"id": c.id, "name": c.name, "config": c.config, "created_at": c.created_at}
            for c in await svc.list_collections()
        ]

    @app.get("/collections/{collection_id}")
    async def get_collection(collection_id: str, svc: RagService = Depends(get_service)):
        coll = await svc.get_collection(collection_id)
        stats = await svc.get_stats(collection_id)
        return {
            "id": coll.id,
            "name": coll.name,
            "config": coll.config,
            "doc_count": stats["doc_count"],
            "chunk_count": stats["chunk_count"],
            "created_at": coll.created_at,
        }

    @app.delete("/collections/{collection_id}", status_code=204)
    async def delete_collection(collection_id: str, svc: RagService = Depends(get_service)):
        await svc.delete_collection(collection_id)

    @app.get("/collections/{collection_id}/stats")
    async def get_stats(collection_id: str, svc: RagService = Depends(get_service)):
        return await svc.get_stats(collection_id)

    @app.post("/collections/{collection_id}/reindex")
    async def reindex(collection_id: str, svc: RagService = Depends(get_service)):
        return {"job_id": await svc.reindex(collection_id)}

    # ---- ingestion -------------------------------------------------------

    @app.post("/collections/{collection_id}/documents")
    async def upsert(
        collection_id: str, body: UpsertRequest, svc: RagService = Depends(get_service)
    ):
        job_id = await svc.upsert(
            collection_id, body.external_id, body.content, body.metadata, body.version_token
        )
        return {"job_id": job_id}

    @app.delete("/collections/{collection_id}/documents/{external_id}")
    async def delete_document(
        collection_id: str, external_id: str, svc: RagService = Depends(get_service)
    ):
        return {"deleted": await svc.delete(collection_id, external_id)}

    @app.post("/collections/{collection_id}/documents/delete-by-filter")
    async def delete_by_filter(
        collection_id: str, body: DeleteByFilterRequest, svc: RagService = Depends(get_service)
    ):
        return {"deleted": await svc.delete_by_filter(collection_id, body.filter)}

    @app.get("/collections/{collection_id}/documents")
    async def list_documents(
        collection_id: str,
        limit: int = 100,
        offset: int = 0,
        svc: RagService = Depends(get_service),
    ):
        return await svc.list_documents(collection_id, None, limit, offset)

    @app.get("/jobs/{job_id}")
    async def job_status(job_id: str, svc: RagService = Depends(get_service)):
        return await svc.get_job_status(job_id)

    # ---- query -----------------------------------------------------------

    @app.post("/query")
    async def query(body: QueryRequest, svc: RagService = Depends(get_service)):
        return await svc.query(
            collection_ids=body.collection_ids,
            text=body.text,
            k=body.k,
            filters=body.filters,
            min_score=body.min_score,
            mode=body.mode,
            principal=body.principal,
            generate=body.generate,
        )

    # ---- ops -------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(svc: RagService = Depends(get_service)):
        try:
            await svc.ready()
        except Exception as exc:  # noqa: BLE001 — surface as 503, fail loud in body
            raise HTTPException(status_code=503, detail=f"not ready: {exc}") from exc
        return {"status": "ready"}
