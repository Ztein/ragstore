"""Weaviate store against a real Weaviate, with real embeddings — tenants, inserts,
hybrid search. No mocks/fakes: vectors come from the real embedding provider."""

import uuid

import pytest_asyncio

APPLE = "apple banana fruit salad sweet"
CARROT = "carrot potato onion vegetable stew"


@pytest_asyncio.fixture
async def vecs(embedder):
    apple, carrot = await embedder.embed([APPLE, CARROT])
    return {"apple": apple, "carrot": carrot}


def _chunk(ext, seq, text, vec, **extra):
    return {"document_external_id": ext, "seq": seq, "text": text, "vector": vec, **extra}


@pytest_asyncio.fixture
async def seeded(wstore, vecs):
    cid = uuid.uuid4().hex
    await wstore.ensure_tenant(cid)
    await wstore.add_chunks(
        cid,
        [
            _chunk("doc-a", 0, APPLE, vecs["apple"],
                   title="Fruit", classification="public", acl_principals=["grp-1"]),
            _chunk("doc-b", 0, CARROT, vecs["carrot"],
                   title="Veg", classification="internal", acl_principals=["grp-2"]),
        ],
    )
    return cid


async def test_ensure_tenant_and_count(wstore, vecs):
    cid = uuid.uuid4().hex
    await wstore.ensure_tenant(cid)
    await wstore.add_chunks(cid, [_chunk("d", 0, APPLE, vecs["apple"])])
    assert await wstore.count_chunks(cid) == 1


async def test_semantic_search_ranks_by_vector(wstore, seeded, vecs):
    results = await wstore.search([seeded], vector=vecs["apple"], text="", k=2, mode="semantic")
    assert results[0]["document_external_id"] == "doc-a"
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["title"] == "Fruit"


async def test_keyword_search_matches_text(wstore, seeded, vecs):
    results = await wstore.search(
        [seeded], vector=vecs["apple"], text="carrot", k=2, mode="keyword"
    )
    assert results[0]["document_external_id"] == "doc-b"


async def test_hybrid_search_returns_scored_chunks(wstore, seeded, vecs):
    results = await wstore.search([seeded], vector=vecs["apple"], text="apple", k=2, mode="hybrid")
    assert results[0]["document_external_id"] == "doc-a"
    assert all("score" in r for r in results)


async def test_filter_by_classification(wstore, seeded, vecs):
    results = await wstore.search(
        [seeded], vector=vecs["apple"], text="", k=5, mode="semantic",
        filters={"classification": "internal"},
    )
    assert {r["document_external_id"] for r in results} == {"doc-b"}


async def test_principal_security_trimming(wstore, seeded, vecs):
    results = await wstore.search(
        [seeded], vector=vecs["carrot"], text="", k=5, mode="semantic", principal="grp-1"
    )
    assert {r["document_external_id"] for r in results} == {"doc-a"}


async def test_min_score_can_return_empty(wstore, seeded, vecs):
    results = await wstore.search(
        [seeded], vector=vecs["apple"], text="", k=5, mode="semantic", min_score=2.0
    )
    assert results == []


async def test_delete_document_chunks(wstore, seeded, vecs):
    await wstore.delete_document_chunks(seeded, "doc-a")
    assert await wstore.count_chunks(seeded) == 1
    results = await wstore.search([seeded], vector=vecs["apple"], text="", k=5, mode="semantic")
    assert {r["document_external_id"] for r in results} == {"doc-b"}


async def test_delete_tenant_drops_everything(wstore, seeded):
    await wstore.delete_tenant(seeded)
    await wstore.ensure_tenant(seeded)
    assert await wstore.count_chunks(seeded) == 0
