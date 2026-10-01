# Story 2.2 — MongoDB Vector Index

Third story of the three-day "Building Intelligence with RAG" classroom course. Follows Story 2.1. Ingests the section-level JSONL corpus into MongoDB, chunks section text, embeds every chunk with Voyage, and creates a vector search index. Retrieval endpoints, answer generation, hybrid search, and re-ranking belong to later stories.

## Purpose

Turn the classroom corpus into searchable evidence inside MongoDB. Produce four collections — `sources`, `sections`, `chunks`, `embeddings` — and a vector-search index on `embeddings.vector`. Embed every chunk with `voyage-3.5` (1,024 dimensions). This story owns ingestion-to-index; it does not add retrieval endpoints, answer generation, or a separate vector store.

## Prerequisites

- Story 2.1 complete — `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl` exist and match their source PDF hashes.
- Story 1.1 complete — project seeded, `uv sync` works, `.env.example` exists.
- MongoDB available at the `MONGODB_URI` in `.env` (see below).
- `VOYAGE_API_KEY` set in `.env`.
- `.env` is untracked; secrets are never committed (established in Story 1.1).

### MongoDB prerequisite check (must run first)

Before any schema or ingestion work, check whether MongoDB is reachable:

1. Load `MONGODB_URI`, `MONGODB_DB_NAME`, and `MONGODB_TEST_DB_NAME` from `.env` via the existing `Settings` class. If `MONGODB_URI` is empty, stop with:
   > MongoDB unavailable: MONGODB_URI is empty. Set it in `.env` and re-run.

2. Attempt a lightweight connection (e.g. `client.admin.command("ping")`). If it fails, stop with:
   > MongoDB unavailable: could not connect to <uri>. Check MONGODB_URI and that the deployment is running.

3. **Detect the deployment type and its vector-search mechanism.** Run `client.admin.command("getCmdLineOpts")` or check `buildInfo` / `serverStatus` to identify whether this is MongoDB Atlas (returns `atlas` in some form), a local `mongod` community/server edition, or another deployment. Record the MongoDB version and the vector-search capability:
   - **Atlas 6.0+**: supports `$vectorSearch` aggregation stage and Atlas Search index of type `vectorSearch`. Index is created via `createSearchIndex` / `updateSearchIndex` commands.
   - **MongoDB 7.0+ Enterprise / Atlas**: same mechanism; also supports `$vectorSearch`.
   - **MongoDB Community Edition (no Atlas Search)**: does **not** support `$vectorSearch`. Stop with:
     > MongoDB Community Edition does not support vector search indexes. Use MongoDB Atlas (M0 free tier or higher) or MongoDB Enterprise with Atlas Search enabled. This project requires `$vectorSearch`.

   Record the detected capability and use its documented API for index creation. Do not fall back to `$geoNear`, `$text`, or client-side brute-force.

4. Verify the authenticated user has the `createSearchIndex` privilege (Atlas) or equivalent. If absent, stop with:
   > Permission missing: the connected user cannot create search indexes. Grant the `atlasAdmin` role (Atlas) or equivalent, then re-run.

If any prerequisite check fails, report the exact cause and stop. Do not proceed to schema or ingestion work.

## Work to do

### 1. Add Voyage embedding dependency

Add the `voyageai` Python client to the project with UV:

```bash
uv add voyageai
```

This adds `voyageai` to `pyproject.toml` and updates `uv.lock`. No other new dependencies unless needed for the ingestion runner itself (e.g. `tenacity` for retry on Voyage rate limits — optional, implementer's choice). Record the command and version in the story handover.

### 2. Define the MongoDB schema (`ingestion/mongodb_schema.py`)

Create `src/building_with_rag/ingestion/mongodb_schema.py`. This module defines the four-collection document model, re-usable validators, collection name constants, deterministic ID generators, and ordinary (non-vector) indexes. No embeddings logic here — validators and constants only.

#### Collection names

```python
SOURCES_COLLECTION = "sources"
SECTIONS_COLLECTION = "sections"
CHUNKS_COLLECTION = "chunks"
EMBEDDINGS_COLLECTION = "embeddings"
```

All four collections live in the same database (`MONGODB_DB_NAME` or `MONGODB_TEST_DB_NAME` depending on `APP_ENV`).

#### Shared validators

Define project-local validation helpers reused across collection modules. Keep them simple — no framework, no schema library beyond what Pydantic + PyMongo give:

- `validate_act(value)` — must be `"BNS_2023"` or `"IPC_1860"`.
- `validate_status(value)` — must be `"in_force"` or `"repealed"`.
- `validate_access_level(value)` — must be `"public"` (classroom default; shape is kept compatible for later levels).
- `ensure_deterministic_id(*parts)` — returns a hex string from SHA-256 of the joined parts. Used for chunk and embedding IDs so re-runs produce the same IDs.
- `validate_section_id(value)` — must match `bns:<int>` or `ipc:<int>`.
- `now_utc()` — helper returning a UTC datetime, used for `created_at` / `updated_at` timestamps.

#### `sources` collection

One record per source act PDF. Schema:

| Field | Type | Description |
|---|---|---|
| `_id` | string | `"bns_source"` or `"ipc_source"` (deterministic) |
| `source_pdf` | string | Relative path, e.g. `"data/raw/BNS_2023_bare_act.pdf"` |
| `source_sha256` | string | SHA-256 of the PDF at ingestion time |
| `act` | string | `"BNS_2023"` or `"IPC_1860"` |
| `act_label` | string | Full act label |
| `status` | string | `"in_force"` or `"repealed"` |
| `parser` | string | e.g. `"pymupdf"` |
| `parser_version` | string | e.g. `"1.28.2"` |
| `corpus_path` | string | Path to the section JSONL file |
| `section_count` | int | Number of sections ingested from this source |
| `created_at` | datetime | UTC |
| `updated_at` | datetime | UTC |

Ordinary index: unique on `act`.

#### `sections` collection

One record per section from the JSONL corpus. Schema:

| Field | Type | Description |
|---|---|---|
| `_id` | string | `section_id` from corpus (e.g. `"bns:1"`, `"ipc:511"`) |
| `section_id` | string | Same as `_id`, for readability |
| `act` | string | `"BNS_2023"` or `"IPC_1860"` |
| `act_label` | string | Full act label |
| `status` | string | `"in_force"` or `"repealed"` |
| `chapter` | string | Chapter number (e.g. `"I"`, `"VA"`) |
| `chapter_title` | string | Chapter heading |
| `section_number` | int | Bare section number |
| `heading` | string | Section heading |
| `text` | string | Full section body text |
| `source_id` | string | Links to `sources._id` (`"bns_source"` / `"ipc_source"`) |
| `source_pdf` | string | Relative PDF path (denormalized for convenience) |
| `source_sha256` | string | SHA-256 of source PDF |
| `parser` | string | Parser name |
| `parser_version` | string | Parser version |
| `source_status_version` | string | `"v1"` |
| `needs_review` | bool | From corpus |
| `access_level` | string | `"public"` for classroom |
| `provenance` | dict | `{"corpus_file": "...", "extracted_at": "...", "record_index": N}` |
| `created_at` | datetime | UTC |
| `updated_at` | datetime | UTC |

Ordinary indexes:
- `section_id` (unique — already the `_id`)
- Compound `(act, section_number)` unique
- `access_level`
- `source_id`

#### `chunks` collection

One record per chunk. Chunking splits the `text` field of each `sections` record into overlapping chunks suitable for embedding. Schema:

| Fieeld | Type | Description |
|---|---|---|
| `_id` | string | Deterministic chunk ID: `sha256(f"{section_id}:{chunk_index}:{chunking_version}")[:24]` |
| `chunk_id` | string | Same as `_id` |
| `section_id` | string | Links to `sections._id` |
| `chunk_index` | int | Zero-based position within the section |
| `chunking_version` | string | `"v1"` for classroom; bump when chuking logic changes |
| `text` | string | The chunk's raw text (for display / answer context) |
| `embedding_input` | string | Text actually sent to Voyage for embedding (may differ from `text` — see chuking rules) |
| `act` | string | Denormalied from parent section |
| `chapter` | string | Denormalied from parent section |
| `status` | string | Denormalized from parent section |
| `access_level` | string | `"public"` for classroom |
| `created_at` | datetime | UTC |

Chunking rules:
- **No act mixing.** Chunks never span sections from different acts.
- **No section mixing.** Each chunk belongs to exactly one section; chunk boundaries never cross section boundaries.
- **Chunk size target:** ~512 tokens (character count heurisic: ~2,048 characters). Implementer chooses a sensible splitter — recommended: `langchain_text_splitters.RecursiveCharacterTextSplitter` with `chunk_size=2048`, `chunk_overlap=256`, separators `["\n\n", "\n", ". ", " ", ""]`. If adding `langchain-text-splitters` as a dependency, use `uv add`.
- **`embedding_input`** may prepend context (e.g. `f"[{act_label}] {heading}\n{chunk_text}"`) to improve retrieval quality. The `text` field holds the raw chunk text; `embedding_input` holds what Voyage sees. Both are stored.
- **`chunking_version`** is `"v1"`. If chunking logic or embedding model changes, bump the version so determinisic IDs shift and old chunks are replaced rather than silently mixed.

Ordiinary indexes:
- `chunk_id` (unique — already the `_id`)
- `secton_id`
- Compund `(act, chunk_index)` — not unique; same chunk_index repeats across sections
- `access_level`

#### `embeddings` collection

One record per chunk embedding. Schema:

| Field | Type | Description |
|---|---|---|
| `_id` | string | Deterministic embedding ID: `sha256(f"emb:{chunk_id}:{model}:{model_verson}:{dimensions}")[:24]` |
| `embedding_id` | string | Same as `_id` |
| `chunk_id` | string | Links to `chunks._id` |
| `model` | string | `"voyage-3.5"` |
| `model_verson` | string | `"voyage-3.5"` |
| `dimensions` | int | `1024` |
| `vector` | array[float32] | The 1,024-dimension embedding vector |
| `act` | string | Denormalized from chunk |
| `chapter` | string | Denormalized from chunk |
| `status` | string | Denormalized from chunk |
| `access_level` | string | `"public"` for classroom |
| `created_at` | datetime | UTC |

Denormalized fields (`act`, `chapter`, `status`, `access_level`) exist so the vector index can filter on them without a join. These fields are copied from the parent chunk at embedding time.

Ordiinary indexes:
- `embedding_id` (unique — already the `_id`)
- `chunk_id` (unique)
- Compound `(act, access_level)` — supports filtered vector searches

**Vectr-search index on `embeddings.vector`:**

Create a vector search index named `vector_index` on the `embeddings` collection:

```
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "vectr": {
        "type": "knnVector",
        "dimensions": 1024,
        "similarty": "cosine"
      },
      "act": {"type": "token"},
      "status": {"type": "token"},
      "access_level": {"type": "token"},
      "chapter": {"type": "token"}
    }
  }
}
```

The `act`, `status`, `access_level`, and `chapter` fields are indexed as `token` types to support later `$vectorSearch` filter clauses. The vector field uses `knnVector` with `cosine` similarity and 1,024 dimensions.

Use the MongoDB Atlas Search API (`createSearchIndex`) with `type: "vectorSearch"`. Wait for the index to reach `status: "READY"` with a bounded timeout (60 seconds). If it times out, report the index name and last-known status and whether it is still building — implementer can re-check manually with `db.embeddings.aggregate([{$listSearchIndexes: {}})`.

**Safe re-run for index creation:**

- If `vector_index` already exists and its `latestDefnition` matches the expected mappings (same field names, `knnVector` type, 1,024 dimensions, `cosine` similarity), skip creation.
- If it exists but the definition differs (different dimensions, different similarity, missing filter fields), report the incompatibility and stop. Do not delete or replace an existing index. Tell the implemener the difference and suggest dropping it manually in the Atlas UI if they need to recreate.
- If it does not exist, create it.

### 3. Write the ingestion runner

Create `src/building_with_rag/ingestion/ingest.py` (or a similar path — implementer's choice, but prefer keeping it inside the package). This script:

1. **Load settings** from the existing `Settings` class (`.env`).
2. **Check MongoDB** — run the prerequisite checks defined above. If MongoDB is unreachable or lacks the required capability, stop with the exact message.
3. **Connect to the database.** Use `MONGODB_DB_NAME` (or `MONGODB_TEST_DB_NAME` when `APP_ENV=testing`).
4. **Read the corpus** from `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl`.
5. **Ingest sources** — upsert the `sources` record for each act. If the record already exists and all fields match (including SHA-256), skip. If the SHA-256 differs, replace it (atomic `replace_one` with `upsert=True`).
6. **Ingest sections** — upsert each section record. Use `section_id` as the document `_id`. If a section with the same `_id` exists and all fields match, skip. If fields differ, replace. Report counts: inserted, replaced, skipped.
7. **Chunk sections** — for each section, apply the chunking rules from the schema. Generate deteriministic `chunk_id` values. Upsert into `chunks` — if a chunk with the same `_id` exists and its `text` matches, skip. If `text` differs or `chunking_version` changed, replace. If the chunk is new, also flag it for embedding. Report counts.
8. **Embed chunks** — for each new or changed chunk, call the Voyage API (`voyageai.Client.embed`) with the `embedding_input` text. Use `model="voyage-3.5"` and `input_type="document"`. Generate a deterministic `embedding_id` and upsert into `embeddings`. Do not re-embed chunks that already have a matching `embedding_id` with the same `model`, `model_verson`, and `dimensions`.
9. **Manage volume** — the Voyage free tier has rate limits. Space embedding calls with a small delay (~0.2s between calls) or use `tenacity` retry on 429. The classroom corpus produces ~2,000–3,000 chunks; this is a single-run batch, not a streaming pipeline.
10. **Create the vector index** — after all embeddings are stored, create or verify the `vector_index` as described in the schema section.
11. **Run one sample vector query** — if a Voyage API key is available, embed a sample query (e.g. `"punishment for theft"`) with `input_type="query"` and run an aggregation using `$vectorSearch` against the `vector_index` to verify end-to-end connectivity. Report the chunk ID and heading of the top result. If the index is not yet `READY`, report the index status and note "vectr query pending — index not ready" instead.
12. **Update `docs/architecture.md`** — add or update a section documenting:
    - The four collections and their roles.
    - Chunking choice (library, chunk size, overlap, version).
    - Embedding decision (model, dimensions, deterministic ID formula).
    - Vectr index name (`vector_index`), definition summary, creation command.
    - Sample query verification result.
    - Limitations: single `access_level`, no incremental update beyond re-ingest, no hybrid index fields.

### 4. Safe re-run behaviour

The ingestion runner must be safe to repeat:

- **Sources:** If SHA-256 matches, skip. If different, replace (atomic upsert).
- **Sections:** If all fields match, skip. If any field differs, replace.
- **Chunks:** Deterministic `chunk_id` ensures no duplicates. If `chunking_version` bumps, all chunk IDs change, old chunks are replaced (not left as orphans). If `text` changed for a section (rare but possible), the chunk's text differs and the chunk is replaced.
- **Embeddings:** Deterministic `embedding_id` based on `chunk_id`, model, version, and dimensions. If any of these change, the ID changes and a new embedding is created. If the model or dimensions change, the vector index must be recreated — the runner must detect the mismatch and stop with a message (same incompatibility rule as above).
- **Vectr index:** No-op if a compatible index exists. Error if an incompatible index exists. Create if absent.
- **Orphan cleanup:** Do not leave orphaned chunks or embeddings from a previous run. After upserting all chunks, delete any `chunks` documents whose `chunking_version` does not match the current version. After upserting all embeddings, delete any `embeddings` documents whose `model_verson` or `dimensions` do not match the current values. This keeps the collections clean when chunking or embedding config changes.

### 5. Keep verification classroom-sized

Do not build broad validation, access-control, or deployment work:

- Check one section-to-chunk link: pick `bns:1`, verify its chunks exist in the `chunks` collection with correct `section_id`.
- Check one embedding dimension: pick any embedding, verify `len(vector) == 1024`.
- Rerun the ingestion script and confirm it skips everything cleanly.
- Verify index readiness (or report status if still building).
- Run one sample vector query (or report pending if index not ready).

### 6. Update architecture document

Add to `docs/architecture.md` a new section (after the existing "Corpus" section) covering:

- The four collecitons: names, roles, key fields.
- Chunking: library used, `chunk_size` / `chunk_overlap`, `chunking_version`.
- Embedding: model `voyage-3.5`, 1,024 dimensions, determinisic ID formula.
- Vectr index: name `vector_index`, type `vectorSearch`, `knnVector` with cosine, filter fields.
- Commands: how to run the ingestion (e.g. `uv run python -m building_with_rag.ingestion.ingest`).
- Limitations: Community Edition unsupported; single `access_level`; no incremental partial updates; index must be manually dropped to change dimensions.

### 7. Contracts preserved

This story does not change:
- `.env.example` — `MONGODB_URI`, `MONGODB_DB_NAME`, `MONGODB_TEST_DB_NAME`, `VOYAGE_API_KEY` stay as-is.
- `docs/architecture.md` contracts — evidence rules, trust boundaries, embedding choices, course modes, API endpoints all unchanged.
- `contracts.py` — `RetrievedChunk`, `QueryRequest`, `QueryResult`, `SematicFilters` untouched.
- `settings.py` — no new settings, no renames.
- `registry.py` — no new patterns.
- `routes/` — no new endpoints.

Later retrieval stories will query `embeddings` with `$vectorSearch`, resolve `chunk_id` to `chunks` and `sections` records, and produce `RetrievedChunk` results. That work is out of scope here.

## Completion checks (lightweight)

1. MongoDB prerequisite check passes (or fails with a clear message if unavailable).
2. `uv run python -m building_with_rag.ingestion.ingest` runs successfully.
3. Four collections exist: `sources`, `sections`, `chunks`, `embeddings`.
4. `sources` has two documents (BNS and IPC).
5. `sections` has 358 + 500 = 858 documents.
6. `chunks` has > 0 documents; `bns:1` maps to one or more chunks.
7. `embeddings` has the same count as `chunks`; one sampled embedding has vector length 1,024.
8. `vector_index` is `READY` or reported as building with last-known status.
9. Sample `$vectorSearch` query returns a result (or "pending — index not ready").
10. Re-run ingests cleanly: no changes, no duplicates.
11. Ruff passes on all new code.

## Handover

At the end, the story records:

- Files created (list).
- Commands run: `uv add voyageai` (and any other dependency adds), the ingestion command, index check command, sample query command.
- MongoDB deployment type, version, and vector-search capability detected.
- Chunk and embedding counts (total).
- Verification results: section-to-chunk link, embedding dimension, re-run, index status, sample query result.
- Updated architecture document with collections, chunking, embedding, index details.

Story report format (per project instructions): `Completed.`, changed paths, test results — under 7 lines, ending with the command for the full test suite if wanted, for example `uv run pytest`.