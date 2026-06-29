"""Weaviate store against a real Weaviate: tenants, BYO-vector inserts, hybrid search."""

import uuid

import pytest_asyncio


def _chunk(ext, seq, text, vec, **extra):
    return {"document_external_id": ext, "seq": seq, "text": text, "vector": vec, **extra}


# 8-dim orthogonal-ish vectors so similarity is unambiguous.
APPLE_VEC = [1.0, 0, 0, 0, 0, 0, 0, 0]
CARROT_VEC = [0, 1.0, 0, 0, 0, 0, 0, 0]


@pytest_asyncio.fixture
async def seeded(wstore):
    cid = uuid.uuid4().hex
    await wstore.ensure_tenant(cid)
    await wstore.add_chunks(
        cid,
        [
            _chunk("doc-a", 0, "apple banana fruit", APPLE_VEC,
                   title="Fruit", classification="public", acl_principals=["grp-1"]),
            _chunk("doc-b", 0, "carrot potato vegetable", CARROT_VEC,
                   title="Veg", classification="internal", acl_principals=["grp-2"]),
        ],
    )
    return cid


async def test_ensure_tenant_and_count(wstore):
    cid = uuid.uuid4().hex
    await wstore.ensure_tenant(cid)
    await wstore.add_chunks(cid, [_chunk("d", 0, "hello", APPLE_VEC)])
    assert await wstore.count_chunks(cid) == 1


async def test_semantic_search_ranks_by_vector(wstore, seeded):
    results = await wstore.search([seeded], vector=APPLE_VEC, text="", k=2, mode="semantic")
    assert results[0]["document_external_id"] == "doc-a"
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["title"] == "Fruit"


async def test_keyword_search_matches_text(wstore, seeded):
    results = await wstore.search([seeded], vector=APPLE_VEC, text="carrot", k=2, mode="keyword")
    assert results[0]["document_external_id"] == "doc-b"


async def test_hybrid_search_returns_scored_chunks(wstore, seeded):
    results = await wstore.search([seeded], vector=APPLE_VEC, text="apple", k=2, mode="hybrid")
    assert results[0]["document_external_id"] == "doc-a"
    assert all("score" in r for r in results)


async def test_filter_by_classification(wstore, seeded):
    results = await wstore.search(
        [seeded], vector=APPLE_VEC, text="", k=5, mode="semantic",
        filters={"classification": "internal"},
    )
    assert {r["document_external_id"] for r in results} == {"doc-b"}


async def test_principal_security_trimming(wstore, seeded):
    results = await wstore.search(
        [seeded], vector=CARROT_VEC, text="", k=5, mode="semantic", principal="grp-1"
    )
    assert {r["document_external_id"] for r in results} == {"doc-a"}


async def test_min_score_can_return_empty(wstore, seeded):
    results = await wstore.search(
        [seeded], vector=APPLE_VEC, text="", k=5, mode="semantic", min_score=2.0
    )
    assert results == []


async def test_delete_document_chunks(wstore, seeded):
    await wstore.delete_document_chunks(seeded, "doc-a")
    assert await wstore.count_chunks(seeded) == 1
    results = await wstore.search([seeded], vector=APPLE_VEC, text="", k=5, mode="semantic")
    assert {r["document_external_id"] for r in results} == {"doc-b"}


async def test_delete_tenant_drops_everything(wstore, seeded):
    await wstore.delete_tenant(seeded)
    await wstore.ensure_tenant(seeded)
    assert await wstore.count_chunks(seeded) == 0
