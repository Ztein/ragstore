# ragstore

Standalone, source-agnostic RAG retrieval component. It owns chunking, embeddings,
the vector + keyword index, collections, retrieval, and optional generation. It does
**not** crawl sources, convert files, or read live documents — that stays in the
consuming application (e.g. doclib).

- **Storage:** Weaviate (vectors + native hybrid search) + embedded SQLite (collection/
  document registry and the async ingestion job queue).
- **Providers:** OpenAI-compatible HTTP for both embeddings and the (optional) generation
  LLM, configured via env — no provider named in code, no local ML models in the image.
- **Transport:** HTTP/JSON API with Bearer auth.

See [`DOCS`](DOCS) / the PRD for the full interface. Run locally with `docker compose up`.

## Status

Early development. Built strictly test-first.
