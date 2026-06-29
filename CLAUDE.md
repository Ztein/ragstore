# ragstore — engineering principles & map

`ragstore` is a standalone, source-agnostic RAG retrieval component. It owns chunking,
embeddings, the vector + keyword index, collections, retrieval, and optional
generation. It does **not** crawl sources, convert files, or read live documents —
that belongs to the consuming application (e.g. doclib).

## Non-negotiable principles

1. **Strict TDD.** Write the failing test first, then the code. Every behavior has a test.
2. **Fail loudly.** Surface errors with clear messages. Never hide a failure or return a
   misleading success. Missing config raises at startup; a failed ingestion job is
   recorded as `failed` with the error string.
3. **Minimal / no fallbacks.** If something is misconfigured or a dependency is down, it
   raises — we do not silently degrade (no "offline embedding mode", no swallowed
   exceptions).
4. **Never mock or fake — test against the real source.** No mocks, no fakes, no stub
   servers, in product code OR tests. Tests run against a **real Weaviate**, a **real
   SQLite file**, and the **real** external embedding/LLM provider. Real credentials are
   required to run the provider tests; provide them via `.env` locally or CI secrets.
   Without credentials those tests skip (they are never replaced by a fake).
5. **External providers only.** Embeddings and the generation LLM are external,
   OpenAI-compatible HTTP endpoints configured via env. No ML models in the image.
6. **Hardened image.** The built image must pass Trivy with **zero fixable HIGH/CRITICAL
   CVEs** (`trivy image --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1`).

## Architecture

- **API** — FastAPI + uvicorn, Bearer auth (`Authorization: Bearer` or `X-API-Key`).
- **Weaviate** — one `Chunk` collection, multi-tenant; each ragstore collection is a
  tenant. `vectorizer: none` (we bring our own vectors). Native hybrid (BM25 + vector).
- **SQLite** — relational source of truth for collections/documents + the durable async
  ingestion job queue (single in-process asyncio worker).
- **Embeddings / LLM** — `httpx` against OpenAI-compatible endpoints. No litellm/torch.

Module map (`src/ragstore/`): `config.py` (fail-loud settings), `sqlite_store.py`,
`weaviate_store.py`, `embeddings.py`, `llm.py`, `chunker.py`, `service.py` (orchestration),
`worker.py` (ingestion loop), `auth.py`, `app.py` (HTTP), `__main__.py` (uvicorn entry).

## Working on this repo

```bash
docker compose up -d weaviate      # start the real Weaviate dependency
cp .env.example .env               # fill in real EMBEDDING_*/LLM_* credentials
uv sync --extra dev                # install (Python 3.13 via uv)
make hooks                         # enable the pre-push hook (lint + format + tests)
uv run pytest                      # full suite (real Weaviate + SQLite + real provider)
uv run ruff check src tests        # lint
make build && make scan            # build image + Trivy gate
```

**Everything runs locally — there is no GitHub Actions CI.** Tests need a real Weaviate
and a real embedding/LLM provider, and we deliberately keep provider credentials off
GitHub. The `pre-push` hook (`make hooks`) runs lint + format + the full suite before
every push; image build + Trivy run via `make build && make scan`.

The suite hits the real embedding/LLM provider — expect occasional transient provider
errors to surface (that's the point: we don't hide them behind a fake). Re-run if a real
dependency hiccups.

Config is environment-driven and fail-loud — see `.env.example`. The component is built
as an image and run as a container; deployment is **not** part of this repo so it can run
in any environment.
