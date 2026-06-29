"""SQLite store: collections, documents, and the job queue against a real DB file."""

import pytest

CFG = {"embedding_model": "m", "embedding_dim": 8, "chunk_size": 500, "chunk_overlap": 100}


async def test_create_get_list_delete_collection(store):
    cid = await store.create_collection("docs", CFG)
    got = await store.get_collection(cid)
    assert got is not None
    assert got.name == "docs"
    assert got.config["embedding_dim"] == 8

    assert (await store.get_collection_by_name("docs")).id == cid
    assert [c.id for c in await store.list_collections()] == [cid]

    assert await store.delete_collection(cid) is True
    assert await store.get_collection(cid) is None
    assert await store.delete_collection(cid) is False


async def test_duplicate_collection_name_fails_loud(store):
    await store.create_collection("docs", CFG)
    with pytest.raises(ValueError, match="already exists"):
        await store.create_collection("docs", CFG)


async def test_put_get_update_document_and_cascade(store):
    cid = await store.create_collection("docs", CFG)
    meta = {"title": "T", "path": "/a/b.md", "classification": "public"}
    await store.put_document(cid, "ext1", "hello world", meta, "v1")

    doc = await store.get_document(cid, "ext1")
    assert doc["content"] == "hello world"
    assert doc["version_token"] == "v1"
    created = doc["created_at"]

    # Update preserves created_at, bumps modified_at + version.
    await store.put_document(cid, "ext1", "hello again", meta, "v2")
    doc2 = await store.get_document(cid, "ext1")
    assert doc2["content"] == "hello again"
    assert doc2["version_token"] == "v2"
    assert doc2["created_at"] == created

    # Deleting the collection cascades to its documents.
    await store.delete_collection(cid)
    assert await store.get_document(cid, "ext1") is None


async def test_list_and_filter_documents(store):
    cid = await store.create_collection("docs", CFG)
    await store.put_document(
        cid, "a", "x", {"path": "/root/one.md", "classification": "public"}, "v"
    )
    await store.put_document(
        cid, "b", "y", {"path": "/root/sub/two.md", "classification": "internal"}, "v"
    )
    await store.put_document(
        cid, "c", "z", {"path": "/other/three.md", "classification": "public"}, "v"
    )

    assert {d["external_id"] for d in await store.list_documents(cid)} == {"a", "b", "c"}
    public = await store.list_documents(cid, filter={"classification": "public"})
    assert {d["external_id"] for d in public} == {"a", "c"}

    under_root = await store.find_documents_by_filter(cid, {"path_prefix": "/root/"})
    assert set(under_root) == {"a", "b"}


async def test_delete_document(store):
    cid = await store.create_collection("docs", CFG)
    await store.put_document(cid, "a", "x", {}, "v")
    assert await store.delete_document(cid, "a") is True
    assert await store.delete_document(cid, "a") is False


async def test_job_queue_lifecycle(store):
    cid = await store.create_collection("docs", CFG)
    job_id = await store.enqueue_job("upsert", cid, "ext1", {"foo": "bar"})

    job = await store.get_job(job_id)
    assert job["state"] == "queued"
    assert job["payload"] == {"foo": "bar"}

    claimed = await store.claim_next_job()
    assert claimed["id"] == job_id
    assert claimed["state"] == "running"
    assert claimed["started_at"] is not None

    # Nothing else queued.
    assert await store.claim_next_job() is None

    await store.finish_job(job_id, "done", chunks=3)
    done = await store.get_job(job_id)
    assert done["state"] == "done"
    assert done["chunks"] == 3
    assert done["finished_at"] is not None


async def test_claim_orders_oldest_first(store):
    cid = await store.create_collection("docs", CFG)
    first = await store.enqueue_job("upsert", cid, "a", {})
    second = await store.enqueue_job("upsert", cid, "b", {})
    assert (await store.claim_next_job())["id"] == first
    assert (await store.claim_next_job())["id"] == second
