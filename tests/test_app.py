"""HTTP API e2e: auth, the full ingest→query→generate flow, error mapping."""

from httpx import ASGITransport, AsyncClient

from ragstore.app import create_app

META = {"title": "Doc A", "source_uri": "https://x/a", "classification": "public"}


async def _ingest(api, worker, cid, ext, content, meta=META, version="v1"):
    r = await api.post(f"/collections/{cid}/documents", json={
        "external_id": ext, "content": content, "metadata": meta, "version_token": version,
    })
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert await worker.run_once() is True
    status = (await api.get(f"/jobs/{job_id}")).json()
    assert status["state"] == "done", status
    return status


async def test_health_and_ready_are_public(service):
    app = create_app(service)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as anon:
        assert (await anon.get("/health")).json() == {"status": "ok"}
        ready = await anon.get("/ready")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready"}


async def test_auth_required(service):
    app = create_app(service)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as anon:
        assert (await anon.get("/collections")).status_code == 401
        bad = await anon.get("/collections", headers={"Authorization": "Bearer wrong"})
        assert bad.status_code == 401


async def test_x_api_key_header_accepted(service):
    app = create_app(service)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.get("/collections", headers={"X-API-Key": "test-key"})).status_code == 200


async def test_create_and_get_collection(api, provider):
    r = await api.post("/collections", json={"name": "docs"})
    assert r.status_code == 201
    cid = r.json()["id"]

    got = await api.get(f"/collections/{cid}")
    assert got.status_code == 200
    body = got.json()
    assert body["name"] == "docs"
    assert body["doc_count"] == 0
    assert body["config"]["embedding_dim"] == provider["embedding_dim"]

    listing = (await api.get("/collections")).json()
    assert [c["id"] for c in listing] == [cid]


async def test_duplicate_name_is_400(api):
    await api.post("/collections", json={"name": "docs"})
    r = await api.post("/collections", json={"name": "docs"})
    assert r.status_code == 400
    assert "already exists" in r.json()["detail"]


async def test_dim_mismatch_is_400(api):
    r = await api.post("/collections", json={"name": "bad", "config": {"embedding_dim": 1536}})
    assert r.status_code == 400


async def test_full_ingest_query_flow(api, worker):
    cid = (await api.post("/collections", json={"name": "docs"})).json()["id"]
    status = await _ingest(api, worker, cid, "a", "apple banana fruit salad")
    assert status["chunks"] >= 1

    stats = (await api.get(f"/collections/{cid}/stats")).json()
    assert stats["doc_count"] == 1 and stats["chunk_count"] >= 1

    r = await api.post("/query", json={
        "collection_ids": [cid], "text": "apple banana fruit salad", "k": 3, "mode": "semantic",
    })
    assert r.status_code == 200
    assert r.json()["chunks"][0]["document_external_id"] == "a"


async def test_query_with_generation(api, worker):
    cid = (await api.post("/collections", json={"name": "docs"})).json()["id"]
    await _ingest(api, worker, cid, "a", "apple banana fruit")
    r = await api.post("/query", json={
        "collection_ids": [cid], "text": "apple", "k": 3, "mode": "semantic",
        "generate": {"instructions": "Be terse."},
    })
    body = r.json()
    assert isinstance(body["answer"], str) and body["answer"].strip()
    assert body["citations"][0]["document_external_id"] == "a"


async def test_list_documents_and_delete(api, worker):
    cid = (await api.post("/collections", json={"name": "docs"})).json()["id"]
    await _ingest(api, worker, cid, "a", "apple banana")

    docs = (await api.get(f"/collections/{cid}/documents")).json()
    assert docs[0]["external_id"] == "a"
    assert "content" not in docs[0]

    d = await api.delete(f"/collections/{cid}/documents/a")
    assert d.json() == {"deleted": True}
    assert (await api.get(f"/collections/{cid}/stats")).json()["doc_count"] == 0


async def test_delete_by_filter_endpoint(api, worker):
    cid = (await api.post("/collections", json={"name": "docs"})).json()["id"]
    await _ingest(api, worker, cid, "a", "x", {"path": "/root/a.md"})
    await _ingest(api, worker, cid, "b", "y", {"path": "/other/b.md"})
    r = await api.post(f"/collections/{cid}/documents/delete-by-filter",
                       json={"filter": {"path_prefix": "/root/"}})
    assert r.json() == {"deleted": ["a"]}


async def test_reindex_endpoint(api, worker):
    cid = (await api.post("/collections", json={"name": "docs"})).json()["id"]
    await _ingest(api, worker, cid, "a", "apple banana")
    job_id = (await api.post(f"/collections/{cid}/reindex")).json()["job_id"]
    await worker.run_once()
    assert (await api.get(f"/jobs/{job_id}")).json()["state"] == "done"


async def test_delete_collection(api):
    cid = (await api.post("/collections", json={"name": "docs"})).json()["id"]
    assert (await api.delete(f"/collections/{cid}")).status_code == 204
    assert (await api.get(f"/collections/{cid}")).status_code == 404


async def test_missing_resources_are_404(api):
    assert (await api.get("/collections/nope")).status_code == 404
    assert (await api.get("/jobs/nope")).status_code == 404
    r = await api.post("/query", json={"collection_ids": ["nope"], "text": "x", "k": 1})
    assert r.status_code == 404
