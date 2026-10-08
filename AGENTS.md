<!-- bmad:context -->
<!-- Verified 2026-10-08 against 7448b9a. Managed by bmad-project-context; edits inside this block are replaced on refresh. Keep anything you want preserved outside the markers. -->

## building-with-rag

Capstone RAG API for BNS/IPC legal questions. Python 3.12+, FastAPI, MongoDB Atlas vector search, Voyage embeddings, uv. Fixed contracts and design choices: `docs/architecture.md`; implemented stories: `docs/stories/`; manual curl tests: `docs/manual-tests.md`.

## Policy

- Before adding a feature or mode, read `docs/architecture.md`; extend contracts additively — never rename fields, simplify them, or add provider-specific variants.
- Never change the embedding model (`voyage-3.5`, 1024 dims) or existing `rag-*` model IDs.
- Never commit secrets; `.env` is untracked, new variables go in `.env.example`.
- The app must start with no database or model credentials; keep `GET /healthz` working without them.

## Where things are

- New retrieval mode: add it to `src/building_with_rag/registry.py`, route it in `pipeline.py`, implement it in `retrieval/<mode>.py`.
- HTTP routes: `src/building_with_rag/routes/`; answer generation: `generation/`.
- Open WebUI client scripts: `open_webui_functions/` (separate client, not part of the API).

## Running and verifying

- Prefix every command with `uv run` (e.g. `uv run pytest tests/test_x.py`, `uv run ruff check`); bare `pytest`/`python` run outside the project environment.
- Keep tests offline: fake Mongo, Voyage and model calls with `monkeypatch`, as existing tests do.
- Start the API: `uv run uvicorn building_with_rag.app:app --host 127.0.0.1 --port 8000`.
- Ingestion takes ~40 min; set `APP_ENV=testing` to target the test database first.
<!-- /bmad:context -->
