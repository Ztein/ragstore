"""Weaviate store: chunks + vectors + native hybrid (BM25 + vector) search.

One ``Chunk`` collection with multi-tenancy enabled; each ragstore collection maps
to a Weaviate tenant (clean isolation and drop-on-delete). We bring our own vectors
(``vectorizer: none``) — the component owns embeddings.

Fail loud: connection and schema problems propagate; no silent degradation.
"""

from __future__ import annotations

import json
from typing import Any

import weaviate
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    StopwordsPreset,
    Tokenization,
    VectorDistances,
)
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.tenants import Tenant

CHUNK_CLASS = "Chunk"

# Exact-match fields use FIELD tokenization so `equal`/`contains` match the whole
# value (otherwise "doc-a" tokenizes on the hyphen and matches "doc-b" too).
# `text`/`title` keep default word tokenization for BM25.
_PROPERTIES = [
    Property(name="document_external_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="seq", data_type=DataType.INT),
    Property(name="text", data_type=DataType.TEXT),
    Property(name="title", data_type=DataType.TEXT),
    Property(name="source_uri", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="location", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="classification", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(
        name="acl_principals", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.FIELD
    ),
    Property(name="metadata", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
]


class WeaviateStore:
    def __init__(
        self,
        http_host: str,
        http_port: int,
        grpc_host: str,
        grpc_port: int,
        http_secure: bool = False,
        grpc_secure: bool = False,
        api_key: str | None = None,
    ) -> None:
        self._kwargs = dict(
            http_host=http_host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=grpc_secure,
        )
        if api_key:
            self._kwargs["auth_credentials"] = weaviate.classes.init.Auth.api_key(api_key)
        self._client: weaviate.WeaviateAsyncClient | None = None

    async def connect(self) -> None:
        self._client = weaviate.use_async_with_custom(**self._kwargs)
        await self._client.connect()
        await self._ensure_schema()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @property
    def client(self) -> weaviate.WeaviateAsyncClient:
        if self._client is None:
            raise RuntimeError("WeaviateStore not connected — call connect() first")
        return self._client

    async def _ensure_schema(self) -> None:
        if not await self.client.collections.exists(CHUNK_CLASS):
            await self.client.collections.create(
                name=CHUNK_CLASS,
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                ),
                # No stopwords: exact-id equal()/delete on text fields must match even
                # when the id is a single stopword token like "a".
                inverted_index_config=Configure.inverted_index(
                    stopwords_preset=StopwordsPreset.NONE
                ),
                multi_tenancy_config=Configure.multi_tenancy(enabled=True),
                properties=_PROPERTIES,
            )

    def _chunks(self):
        return self.client.collections.get(CHUNK_CLASS)

    # ---- tenants (= ragstore collections) --------------------------------

    async def ensure_tenant(self, collection_id: str) -> None:
        coll = self._chunks()
        existing = await coll.tenants.get()
        if collection_id not in existing:
            await coll.tenants.create([Tenant(name=collection_id)])

    async def delete_tenant(self, collection_id: str) -> None:
        coll = self._chunks()
        existing = await coll.tenants.get()
        if collection_id in existing:
            await coll.tenants.remove([collection_id])

    # ---- chunks ----------------------------------------------------------

    async def add_chunks(self, collection_id: str, chunks: list[dict[str, Any]]) -> int:
        """Insert chunk objects (each carries its own vector) into the tenant."""
        tcoll = self._chunks().with_tenant(collection_id)
        objects = [
            DataObject(
                properties={
                    "document_external_id": c["document_external_id"],
                    "seq": c["seq"],
                    "text": c["text"],
                    "title": c.get("title") or "",
                    "source_uri": c.get("source_uri") or "",
                    "location": json.dumps(c.get("location") or {}),
                    "classification": c.get("classification") or "",
                    "acl_principals": c.get("acl_principals") or [],
                    "metadata": json.dumps(c.get("metadata") or {}),
                },
                vector=c["vector"],
            )
            for c in chunks
        ]
        result = await tcoll.data.insert_many(objects)
        if result.has_errors:
            raise RuntimeError(f"Weaviate insert errors: {result.errors}")
        return len(objects)

    async def delete_document_chunks(self, collection_id: str, external_id: str) -> None:
        tcoll = self._chunks().with_tenant(collection_id)
        await tcoll.data.delete_many(
            where=Filter.by_property("document_external_id").equal(external_id)
        )

    async def count_chunks(self, collection_id: str) -> int:
        tcoll = self._chunks().with_tenant(collection_id)
        result = await tcoll.aggregate.over_all(total_count=True)
        return result.total_count

    # ---- search ----------------------------------------------------------

    async def search(
        self,
        collection_ids: list[str],
        vector: list[float],
        text: str,
        k: int,
        mode: str = "hybrid",
        filters: dict[str, Any] | None = None,
        min_score: float | None = None,
        principal: str | None = None,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        where = self._build_filter(filters, principal)
        merged: list[dict[str, Any]] = []
        for cid in collection_ids:
            tcoll = self._chunks().with_tenant(cid)
            if mode == "semantic":
                res = await tcoll.query.near_vector(
                    near_vector=vector,
                    limit=k,
                    filters=where,
                    return_metadata=MetadataQuery(distance=True),
                )
            elif mode == "keyword":
                res = await tcoll.query.bm25(
                    query=text,
                    limit=k,
                    filters=where,
                    return_metadata=MetadataQuery(score=True),
                )
            elif mode == "hybrid":
                res = await tcoll.query.hybrid(
                    query=text,
                    vector=vector,
                    alpha=alpha,
                    limit=k,
                    filters=where,
                    return_metadata=MetadataQuery(score=True),
                )
            else:
                raise ValueError(f"unknown query mode: {mode!r}")
            merged.extend(self._to_results(cid, res.objects, mode))

        merged.sort(key=lambda r: r["score"], reverse=True)
        if min_score is not None:
            merged = [r for r in merged if r["score"] >= min_score]
        return merged[:k]

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None, principal: str | None):
        clauses = []
        if filters:
            for key, value in filters.items():
                if key in ("classification", "document_external_id"):
                    clauses.append(Filter.by_property(key).equal(value))
        if principal:
            clauses.append(Filter.by_property("acl_principals").contains_any([principal]))
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else Filter.all_of(clauses)

    @staticmethod
    def _to_results(collection_id: str, objects: list[Any], mode: str) -> list[dict[str, Any]]:
        out = []
        for obj in objects:
            props = obj.properties
            if mode == "semantic":
                score = 1.0 - (obj.metadata.distance or 0.0)
            else:
                score = obj.metadata.score or 0.0
            out.append(
                {
                    "collection_id": collection_id,
                    "text": props["text"],
                    "score": float(score),
                    "document_external_id": props["document_external_id"],
                    "title": props.get("title") or None,
                    "source_uri": props.get("source_uri") or None,
                    "location": json.loads(props.get("location") or "{}"),
                    "metadata": json.loads(props.get("metadata") or "{}"),
                }
            )
        return out
