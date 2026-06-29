"""SQLite store: the relational source of truth for collections and documents, plus
the durable async ingestion job queue.

Single-process, single-worker design — one connection, WAL mode, foreign keys on.
Fail loud: integrity violations (e.g. duplicate collection name) propagate as errors.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiosqlite

JobState = str  # "queued" | "running" | "done" | "failed"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    config      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    collection_id TEXT NOT NULL,
    external_id   TEXT NOT NULL,
    version_token TEXT,
    content       TEXT NOT NULL,
    metadata      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    modified_at   TEXT NOT NULL,
    PRIMARY KEY (collection_id, external_id),
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    external_id   TEXT,
    state         TEXT NOT NULL,
    error         TEXT,
    chunks        INTEGER,
    payload       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, created_at);
"""


@dataclass
class Collection:
    id: str
    name: str
    config: dict[str, Any]
    created_at: str


class SqliteStore:
    def __init__(self, path: str) -> None:
        self._path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SqliteStore not connected — call connect() first")
        return self._db

    # ---- collections -----------------------------------------------------

    async def create_collection(self, name: str, config: dict[str, Any]) -> str:
        cid = _new_id()
        try:
            await self.db.execute(
                "INSERT INTO collections (id, name, config, created_at) VALUES (?, ?, ?, ?)",
                (cid, name, json.dumps(config), _now()),
            )
            await self.db.commit()
        except aiosqlite.IntegrityError as exc:
            raise ValueError(f"collection name already exists: {name!r}") from exc
        return cid

    async def get_collection(self, collection_id: str) -> Collection | None:
        cur = await self.db.execute(
            "SELECT id, name, config, created_at FROM collections WHERE id = ?",
            (collection_id,),
        )
        row = await cur.fetchone()
        return self._row_to_collection(row) if row else None

    async def get_collection_by_name(self, name: str) -> Collection | None:
        cur = await self.db.execute(
            "SELECT id, name, config, created_at FROM collections WHERE name = ?",
            (name,),
        )
        row = await cur.fetchone()
        return self._row_to_collection(row) if row else None

    async def list_collections(self) -> list[Collection]:
        cur = await self.db.execute(
            "SELECT id, name, config, created_at FROM collections ORDER BY created_at"
        )
        return [self._row_to_collection(r) for r in await cur.fetchall()]

    async def delete_collection(self, collection_id: str) -> bool:
        cur = await self.db.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
        await self.db.commit()
        return cur.rowcount > 0

    async def collection_stats(self, collection_id: str) -> dict[str, Any]:
        cur = await self.db.execute(
            "SELECT COUNT(*) AS doc_count, MAX(modified_at) AS last_ingested_at "
            "FROM documents WHERE collection_id = ?",
            (collection_id,),
        )
        row = await cur.fetchone()
        return {"doc_count": row["doc_count"], "last_ingested_at": row["last_ingested_at"]}

    @staticmethod
    def _row_to_collection(row: aiosqlite.Row) -> Collection:
        return Collection(
            id=row["id"],
            name=row["name"],
            config=json.loads(row["config"]),
            created_at=row["created_at"],
        )

    # ---- documents -------------------------------------------------------

    async def get_document(self, collection_id: str, external_id: str) -> dict[str, Any] | None:
        cur = await self.db.execute(
            "SELECT collection_id, external_id, version_token, content, metadata, "
            "created_at, modified_at FROM documents WHERE collection_id = ? AND external_id = ?",
            (collection_id, external_id),
        )
        row = await cur.fetchone()
        return self._row_to_document(row) if row else None

    async def put_document(
        self,
        collection_id: str,
        external_id: str,
        content: str,
        metadata: dict[str, Any],
        version_token: str | None,
    ) -> None:
        now = _now()
        existing = await self.get_document(collection_id, external_id)
        created_at = existing["created_at"] if existing else now
        await self.db.execute(
            "INSERT INTO documents (collection_id, external_id, version_token, content, "
            "metadata, created_at, modified_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(collection_id, external_id) DO UPDATE SET "
            "version_token=excluded.version_token, content=excluded.content, "
            "metadata=excluded.metadata, modified_at=excluded.modified_at",
            (
                collection_id,
                external_id,
                version_token,
                content,
                json.dumps(metadata),
                created_at,
                now,
            ),
        )
        await self.db.commit()

    async def delete_document(self, collection_id: str, external_id: str) -> bool:
        cur = await self.db.execute(
            "DELETE FROM documents WHERE collection_id = ? AND external_id = ?",
            (collection_id, external_id),
        )
        await self.db.commit()
        return cur.rowcount > 0

    async def list_documents(
        self,
        collection_id: str,
        filter: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        cur = await self.db.execute(
            "SELECT collection_id, external_id, version_token, content, metadata, "
            "created_at, modified_at FROM documents WHERE collection_id = ? "
            "ORDER BY created_at LIMIT ? OFFSET ?",
            (collection_id, limit, offset),
        )
        docs = [self._row_to_document(r) for r in await cur.fetchall()]
        return [d for d in docs if _matches(d["metadata"], filter)]

    async def find_documents_by_filter(
        self, collection_id: str, filter: dict[str, Any]
    ) -> list[str]:
        cur = await self.db.execute(
            "SELECT external_id, metadata FROM documents WHERE collection_id = ?",
            (collection_id,),
        )
        return [
            r["external_id"]
            for r in await cur.fetchall()
            if _matches(json.loads(r["metadata"]), filter)
        ]

    @staticmethod
    def _row_to_document(row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "collection_id": row["collection_id"],
            "external_id": row["external_id"],
            "version_token": row["version_token"],
            "content": row["content"],
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
            "modified_at": row["modified_at"],
        }

    # ---- jobs ------------------------------------------------------------

    async def enqueue_job(
        self,
        type: str,
        collection_id: str,
        external_id: str | None,
        payload: dict[str, Any],
    ) -> str:
        job_id = _new_id()
        await self.db.execute(
            "INSERT INTO jobs (id, type, collection_id, external_id, state, payload, created_at) "
            "VALUES (?, ?, ?, ?, 'queued', ?, ?)",
            (job_id, type, collection_id, external_id, json.dumps(payload), _now()),
        )
        await self.db.commit()
        return job_id

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        cur = await self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cur.fetchone()
        return self._row_to_job(row) if row else None

    async def claim_next_job(self) -> dict[str, Any] | None:
        """Atomically move the oldest queued job to running and return it."""
        await self.db.execute("BEGIN IMMEDIATE")
        try:
            cur = await self.db.execute(
                "SELECT id FROM jobs WHERE state = 'queued' ORDER BY created_at LIMIT 1"
            )
            row = await cur.fetchone()
            if row is None:
                await self.db.execute("COMMIT")
                return None
            await self.db.execute(
                "UPDATE jobs SET state = 'running', started_at = ? WHERE id = ?",
                (_now(), row["id"]),
            )
            await self.db.execute("COMMIT")
        except BaseException:
            await self.db.execute("ROLLBACK")
            raise
        return await self.get_job(row["id"])

    async def finish_job(
        self,
        job_id: str,
        state: JobState,
        error: str | None = None,
        chunks: int | None = None,
    ) -> None:
        await self.db.execute(
            "UPDATE jobs SET state = ?, error = ?, chunks = ?, finished_at = ? WHERE id = ?",
            (state, error, chunks, _now(), job_id),
        )
        await self.db.commit()

    @staticmethod
    def _row_to_job(row: aiosqlite.Row) -> dict[str, Any]:
        job = dict(row)
        job["payload"] = json.loads(job["payload"])
        return job


def _matches(metadata: dict[str, Any], filter: dict[str, Any] | None) -> bool:
    """Minimal v1 metadata predicate: exact-match on keys, plus `path_prefix`."""
    if not filter:
        return True
    for key, value in filter.items():
        if key == "path_prefix":
            path = metadata.get("path") or ""
            if not isinstance(path, str) or not path.startswith(value):
                return False
        elif metadata.get(key) != value:
            return False
    return True
