"""Service-level e2e: ingestion via the worker, query, generation — real stores."""

import pytest

from ragstore.service import NotFoundError

META = {"title": "Doc A", "source_uri": "https://x/a", "path": "/a.md", "classification": "public"}


async def _ingest(service, worker, cid, ext, content, meta=META, version="v1"):
    job_id = await service.upsert(cid, ext, content, meta, version)
    assert await worker.run_once() is True
    status = await service.get_job_status(job_id)
    assert status["state"] == "done", status
    return status


async def test_create_collection_and_stats(service):
    cid = await service.create_collection("docs")
    coll = await service.get_collection(cid)
    assert coll.name == "docs"
    assert coll.config["embedding_dim"] == service.settings.embedding_dim
    stats = await service.get_stats(cid)
    assert stats == {
        "id": cid,
        "name": "docs",
        "doc_count": 0,
        "chunk_count": 0,
        "last_ingested_at": None,
    }


async def test_dim_mismatch_fails_loud(service):
    with pytest.raises(ValueError, match="embedding_dim"):
        await service.create_collection("bad", {"embedding_dim": 1536})


async def test_upsert_indexes_and_query_finds(service, worker):
    cid = await service.create_collection("docs")
    status = await _ingest(service, worker, cid, "a", "apple banana fruit salad")
    assert status["chunks"] >= 1

    stats = await service.get_stats(cid)
    assert stats["doc_count"] == 1
    assert stats["chunk_count"] >= 1

    result = await service.query([cid], "apple banana fruit salad", k=3, mode="semantic")
    assert result["chunks"][0]["document_external_id"] == "a"


async def test_unchanged_version_is_noop(service, worker):
    cid = await service.create_collection("docs")
    await _ingest(service, worker, cid, "a", "hello world", version="v1")
    before = (await service.get_stats(cid))["chunk_count"]

    # Same version_token → no re-chunk/re-embed.
    job_id = await service.upsert(cid, "a", "totally different content", META, "v1")
    await worker.run_once()
    assert (await service.get_job_status(job_id))["state"] == "done"
    after = (await service.get_stats(cid))["chunk_count"]
    assert after == before
    # Stored content unchanged.
    doc = await service.sqlite.get_document(cid, "a")
    assert doc["content"] == "hello world"


async def test_changed_version_reindexes(service, worker):
    cid = await service.create_collection("docs")
    await _ingest(service, worker, cid, "a", "hello world", version="v1")
    await _ingest(service, worker, cid, "a", "carrot potato stew", version="v2")
    doc = await service.sqlite.get_document(cid, "a")
    assert doc["content"] == "carrot potato stew"
    result = await service.query([cid], "carrot potato stew", k=3, mode="semantic")
    assert result["chunks"][0]["document_external_id"] == "a"


async def test_delete_document(service, worker):
    cid = await service.create_collection("docs")
    await _ingest(service, worker, cid, "a", "apple banana")
    assert await service.delete(cid, "a") is True
    assert (await service.get_stats(cid))["chunk_count"] == 0
    assert (await service.get_stats(cid))["doc_count"] == 0


async def test_delete_by_filter(service, worker):
    cid = await service.create_collection("docs")
    await _ingest(service, worker, cid, "a", "x", {"path": "/root/a.md"})
    await _ingest(service, worker, cid, "b", "y", {"path": "/root/sub/b.md"})
    await _ingest(service, worker, cid, "c", "z", {"path": "/other/c.md"})
    deleted = await service.delete_by_filter(cid, {"path_prefix": "/root/"})
    assert set(deleted) == {"a", "b"}
    assert (await service.get_stats(cid))["doc_count"] == 1


async def test_list_documents_metadata_only(service, worker):
    cid = await service.create_collection("docs")
    await _ingest(service, worker, cid, "a", "apple banana")
    docs = await service.list_documents(cid)
    assert len(docs) == 1
    assert docs[0]["external_id"] == "a"
    assert "content" not in docs[0]
    assert docs[0]["metadata"]["title"] == "Doc A"


async def test_reindex(service, worker):
    cid = await service.create_collection("docs")
    await _ingest(service, worker, cid, "a", "apple banana")
    job_id = await service.reindex(cid)
    assert await worker.run_once() is True
    assert (await service.get_job_status(job_id))["state"] == "done"
    assert (await service.get_stats(cid))["chunk_count"] >= 1


async def test_min_score_honest_empty(service, worker):
    cid = await service.create_collection("docs")
    await _ingest(service, worker, cid, "a", "apple banana")
    result = await service.query([cid], "apple banana", k=3, mode="semantic", min_score=2.0)
    assert result["chunks"] == []


async def test_query_with_generation(service, worker):
    cid = await service.create_collection("docs")
    await _ingest(service, worker, cid, "a", "apple banana fruit")
    result = await service.query(
        [cid], "apple", k=3, mode="semantic", generate={"instructions": "Be terse."}
    )
    assert isinstance(result["answer"], str) and result["answer"].strip()
    assert result["citations"][0]["document_external_id"] == "a"
    assert result["chunks"]


async def test_unknown_collection_and_job_fail_loud(service):
    with pytest.raises(NotFoundError):
        await service.get_collection("nope")
    with pytest.raises(NotFoundError):
        await service.get_job_status("nope")
    with pytest.raises(NotFoundError):
        await service.query(["nope"], "x", k=1)


async def test_failed_job_records_error(service, worker, monkeypatch):
    cid = await service.create_collection("docs")

    async def boom(*a, **k):
        raise RuntimeError("embed exploded")

    monkeypatch.setattr(service.embeddings, "embed", boom)
    job_id = await service.upsert(cid, "a", "apple banana", META, "v1")
    await worker.run_once()
    status = await service.get_job_status(job_id)
    assert status["state"] == "failed"
    assert "embed exploded" in status["error"]
