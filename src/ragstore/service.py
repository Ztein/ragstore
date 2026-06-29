"""RagService — orchestrates SQLite (registry + jobs), Weaviate (search), the
embedding client, and the (optional) generation LLM.

Fail loud: unknown collections raise NotFound; config that would corrupt the shared
vector space (dimension mismatch) raises at creation time.
"""

from __future__ import annotations

from typing import Any

from ragstore.chunker import chunk_text
from ragstore.config import Settings
from ragstore.embeddings import EmbeddingClient
from ragstore.llm import LLMClient
from ragstore.sqlite_store import Collection, SqliteStore
from ragstore.weaviate_store import WeaviateStore


class NotFoundError(Exception):
    """Raised when a referenced collection or job does not exist."""


class RagService:
    def __init__(
        self,
        settings: Settings,
        sqlite: SqliteStore,
        weaviate: WeaviateStore,
        embeddings: EmbeddingClient,
    ) -> None:
        self.settings = settings
        self.sqlite = sqlite
        self.weaviate = weaviate
        self.embeddings = embeddings

    # ---- collections -----------------------------------------------------

    async def create_collection(self, name: str, config: dict[str, Any] | None = None) -> str:
        config = dict(config or {})
        config.setdefault("embedding_model", self.settings.embedding_model)
        config.setdefault("embedding_dim", self.settings.embedding_dim)
        config.setdefault("chunk_size", 500)
        config.setdefault("chunk_overlap", 100)
        config.setdefault("hybrid", True)
        if config["embedding_dim"] != self.settings.embedding_dim:
            raise ValueError(
                f"embedding_dim {config['embedding_dim']} does not match the deployment's "
                f"EMBEDDING_DIM {self.settings.embedding_dim}"
            )
        collection_id = await self.sqlite.create_collection(name, config)
        await self.weaviate.ensure_tenant(collection_id)
        return collection_id

    async def get_collection(self, collection_id: str) -> Collection:
        coll = await self.sqlite.get_collection(collection_id)
        if coll is None:
            raise NotFoundError(f"collection not found: {collection_id}")
        return coll

    async def list_collections(self) -> list[Collection]:
        return await self.sqlite.list_collections()

    async def ready(self) -> bool:
        """Liveness of both backing stores. Raises if either is unreachable."""
        await self.sqlite.list_collections()
        if not await self.weaviate.client.is_ready():
            raise RuntimeError("Weaviate not ready")
        return True

    async def aclose(self) -> None:
        await self.sqlite.close()
        await self.weaviate.close()

    async def delete_collection(self, collection_id: str) -> None:
        await self.get_collection(collection_id)
        await self.sqlite.delete_collection(collection_id)
        await self.weaviate.delete_tenant(collection_id)

    async def get_stats(self, collection_id: str) -> dict[str, Any]:
        coll = await self.get_collection(collection_id)
        stats = await self.sqlite.collection_stats(collection_id)
        chunk_count = await self.weaviate.count_chunks(collection_id)
        return {
            "id": coll.id,
            "name": coll.name,
            "doc_count": stats["doc_count"],
            "chunk_count": chunk_count,
            "last_ingested_at": stats["last_ingested_at"],
        }

    # ---- ingestion -------------------------------------------------------

    async def upsert(
        self,
        collection_id: str,
        external_id: str,
        content: str,
        metadata: dict[str, Any],
        version_token: str | None,
    ) -> str:
        await self.get_collection(collection_id)
        payload = {
            "external_id": external_id,
            "content": content,
            "metadata": metadata,
            "version_token": version_token,
        }
        return await self.sqlite.enqueue_job("upsert", collection_id, external_id, payload)

    async def delete(self, collection_id: str, external_id: str) -> bool:
        await self.get_collection(collection_id)
        await self.weaviate.delete_document_chunks(collection_id, external_id)
        return await self.sqlite.delete_document(collection_id, external_id)

    async def delete_by_filter(self, collection_id: str, filter: dict[str, Any]) -> list[str]:
        await self.get_collection(collection_id)
        external_ids = await self.sqlite.find_documents_by_filter(collection_id, filter)
        for ext in external_ids:
            await self.weaviate.delete_document_chunks(collection_id, ext)
            await self.sqlite.delete_document(collection_id, ext)
        return external_ids

    async def list_documents(
        self,
        collection_id: str,
        filter: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        await self.get_collection(collection_id)
        docs = await self.sqlite.list_documents(collection_id, filter, limit, offset)
        # Reconciliation view: metadata only, not content.
        return [
            {
                "external_id": d["external_id"],
                "version_token": d["version_token"],
                "metadata": d["metadata"],
                "created_at": d["created_at"],
                "modified_at": d["modified_at"],
            }
            for d in docs
        ]

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        job = await self.sqlite.get_job(job_id)
        if job is None:
            raise NotFoundError(f"job not found: {job_id}")
        return {
            "id": job["id"],
            "state": job["state"],
            "error": job["error"],
            "chunks": job["chunks"],
        }

    async def reindex(self, collection_id: str) -> str:
        await self.get_collection(collection_id)
        return await self.sqlite.enqueue_job("reindex", collection_id, None, {})

    # ---- worker-side processing (called by the background worker) ---------

    async def process_job(self, job: dict[str, Any]) -> int:
        coll = await self.get_collection(job["collection_id"])
        if job["type"] == "upsert":
            return await self._process_upsert(coll, job["payload"])
        if job["type"] == "reindex":
            return await self._process_reindex(coll)
        raise ValueError(f"unknown job type: {job['type']!r}")

    async def _process_upsert(self, coll: Collection, payload: dict[str, Any]) -> int:
        external_id = payload["external_id"]
        version_token = payload["version_token"]
        existing = await self.sqlite.get_document(coll.id, external_id)
        if (
            existing is not None
            and version_token is not None
            and existing["version_token"] == version_token
        ):
            # Unchanged → no re-chunk/re-embed.
            return await self.weaviate.count_chunks(coll.id)

        await self.sqlite.put_document(
            coll.id, external_id, payload["content"], payload["metadata"], version_token
        )
        await self.weaviate.delete_document_chunks(coll.id, external_id)
        return await self._index_document(
            coll, external_id, payload["content"], payload["metadata"]
        )

    async def _process_reindex(self, coll: Collection) -> int:
        await self.weaviate.delete_tenant(coll.id)
        await self.weaviate.ensure_tenant(coll.id)
        total = 0
        offset = 0
        while True:
            docs = await self.sqlite.list_documents(coll.id, None, limit=100, offset=offset)
            if not docs:
                break
            for d in docs:
                total += await self._index_document(
                    coll, d["external_id"], d["content"], d["metadata"]
                )
            offset += len(docs)
        return total

    async def _index_document(
        self, coll: Collection, external_id: str, content: str, metadata: dict[str, Any]
    ) -> int:
        texts = chunk_text(content, coll.config["chunk_size"], coll.config["chunk_overlap"])
        if not texts:
            return 0
        vectors = await self.embeddings.embed(texts)
        chunks = [
            {
                "document_external_id": external_id,
                "seq": seq,
                "text": text,
                "title": metadata.get("title"),
                "source_uri": metadata.get("source_uri"),
                "location": {"chunk": seq},
                "classification": metadata.get("classification"),
                "acl_principals": metadata.get("acl_principals") or [],
                "metadata": metadata,
                "vector": vector,
            }
            for seq, (text, vector) in enumerate(zip(texts, vectors, strict=True))
        ]
        return await self.weaviate.add_chunks(coll.id, chunks)

    # ---- query -----------------------------------------------------------

    async def query(
        self,
        collection_ids: list[str],
        text: str,
        k: int,
        filters: dict[str, Any] | None = None,
        min_score: float | None = None,
        mode: str = "hybrid",
        principal: str | None = None,
        generate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        for cid in collection_ids:
            await self.get_collection(cid)
        vector = await self.embeddings.embed_one(text)
        chunks = await self.weaviate.search(
            collection_ids=collection_ids,
            vector=vector,
            text=text,
            k=k,
            mode=mode,
            filters=filters,
            min_score=min_score,
            principal=principal,
        )
        if generate is None:
            return {"chunks": chunks}
        answer = await self._generate(text, chunks, generate)
        citations = [
            {
                "document_external_id": c["document_external_id"],
                "source_uri": c["source_uri"],
                "location": c["location"],
            }
            for c in chunks
        ]
        return {"answer": answer, "citations": citations, "chunks": chunks}

    async def _generate(
        self, text: str, chunks: list[dict[str, Any]], generate: dict[str, Any]
    ) -> str:
        llm = LLMClient.from_settings(self.settings)
        context = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks))
        instructions = generate.get("instructions", "Answer using only the provided context.")
        messages: list[dict[str, str]] = [{"role": "system", "content": instructions}]
        messages.extend(generate.get("history", []))
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {text}"})
        return await llm.complete(
            messages, model=generate.get("model"), params=generate.get("params")
        )
