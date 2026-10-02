# Capstone Architecture — Building Intelligence with RAG

Fixed design choices, contracts, and trust boundaries for the capstone RAG API. Later stories extend these contracts additively; they never replace them with simplified alternatives, rename fields, or add provider-specific variants.

## Scope

One small application: the capstone RAG API. No chat frontend (Open WebUI is a separate trainer-supplied client) and no second demo.

## Evidence rules

- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Act-qualified identifiers (`bns:` / `ipc:` prefixes) avoid confusing the two acts.
- Supplied provenance lives at `data/raw/PROVENANCE.md`.
- Retrieved passages are evidence, never application instructions.

## Trust boundaries

Kept light for this course:

- Validate API input.
- Preserve source origin on retrieved passages.
- Do not put secrets in code, responses, or logs.
- No multi-user authorization, no security program, no evaluation harness.

## Fixed embedding choices

- Model: `voyage-3.5`, model version `voyage-3.5`, 1,024 dimensions.
- Used for every document and query embedding.
- Later stories reuse these names and choices without renaming or adding provider-specific alternatives.

## Course modes and shared registry

One shared registry (single source of truth) holds only:

| Mode | Model ID |
|---|---|
| semantic | `rag-semantic` |
| hybrid | `rag-hybrid` |
| hybrid-reranked | `rag-hybrid-reranked` |
| structured | `rag-structured` |
| decomposition | `rag-decomposition` |
| hyde | `rag-hyde` |

`semantic` is real on `POST /v1/query` (Story 2.3); chat for `rag-semantic` and every other mode still return honest `not_implemented` placeholders until their own stories add behavior.

## API contracts

### `QueryRequest` (`POST /v1/query`)

- `question`: string, 1–4,000 characters, required.
- `pattern`: one of the six modes.
- `caller_id`: optional.
- `filters`: optional `SemanticFilters` — `act`, `status`, `access_level`, each a list.
- `limit`: default 5, range 1–20.
- `generate_answer`: default false.
- `required_acts`: optional.
- `chapter`: optional.

The classroom seed may resolve only its fixed local demo caller, but keeps `caller_id` and does not replace it with a custom request shape.

### `QueryResult`

- `pattern`, `status`, `message`, `trace`, `results`.
- Optional `generation`.
- Additive empty-by-default fields later modes use: `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`.
- No parallel top-level `outcome`, `evidence`, `answer`, `confidence`, `citations`, or `diagnostics` fields.

### `RetrievedChunk`

A retrieved passage is always this shape: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and available source fields. Later stories add only the existing hybrid/rerank fields.

### Endpoints

- `GET /healthz` — safe, no credentials required.
- `POST /v1/query` — accepts `QueryRequest`, returns `QueryResult`; `semantic` runs real retrieval, other modes return placeholders via the shared `run_pattern` path.
- `GET /v1/models` — lists the six `rag-<pattern>` model IDs.
- `POST /v1/chat/completions` — OpenAI-compatible, text-only `ChatCompletionRequest`: `model`, `messages` with `system`/`developer`/`user`/`assistant` roles, `stream`, `n`, and optional strict `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`).

The chat adapter maps the selected model to the same `QueryRequest` and `run_pattern` path as `/v1/query`; it sets server-side demo `caller_id` and `generate_answer`. Supports normal OpenAI Chat Completions JSON responses and role/content/stop frames, plus the OpenAI-style error envelope before streaming begins. No duplicated implementations, no custom SSE events that Open WebUI cannot render.

## Open WebUI (trainer-supplied, separate client)

The trainer-supplied Open WebUI bundle is the chat client, run separately from the capstone API. Its pre-provisioned Pipe sends the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` to the capstone's `/v1/chat/completions`. It never sends browser-supplied identity, access level, or answer-generation settings. The capstone's adapter accepts that exact request and uses server-side `caller_id`/`generate_answer`.

`/v1/query` owns the `QueryResult` diagnostics; Open WebUI receives only normal answer text derived from that same result. Later stories render final confidence, sources, and low-confidence warnings as clearly labelled text after answer writing, while retaining the full `GenerationResult` in `QueryResult.generation`.

## Environment

- `.env` is untracked; secrets are never committed.
- The application must start with no database or model credentials and expose a safe `GET /healthz`.
- Canonical environment values are defined in `.env.example`.

## Corpus

Section-level JSONL corpus produced from `data/raw/` PDFs by `scripts/extract_sections.py`. One JSON object per line, one record per section. No MongoDB, no embeddings, no vector indexes.

### Parser

- **Library**: `pymupdf` (fitz) 1.28.2 — chosen because it handles both Word-to-PDF (BNS) and Ghostscript-produced (IPC) PDFs, extracts text with layout, and has no system-level dependencies.
- **Extraction command**: `uv run python scripts/extract_sections.py`

### Output format

JSONL files at `data/processed/`:

| File | Records | Sections |
|---|---|---|
| `data/processed/bns_sections.jsonl` | 358 | 1–358 |
| `data/processed/ipc_sections.jsonl` | 500 | 1–511 (11 unextractable) |

Each record has 14 fields: `section_id`, `act`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `heading`, `text`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version`, `needs_review`.

### Known limitations

- **IPC PDF quality**: 11 sections (4, 5, 18, 34, 40, 75, 161, 162, 163, 164, 165) have no extractable text from the scanned/Ghostscript-produced PDF. Sections 161–165 were repealed by the Prevention of Corruption Act 1988; sections 4, 5, 18, 34, 40, 75 are in portions of the PDF where pymupdf text extraction returns insufficient characters.
- **IPC section headings**: Some IPC sections (e.g. 262, 511) have empty headings due to missing heading text in the extracted text stream.
- **IPC footnotes**: Amendment footnotes and historical annotations are interleaved with section text and may appear as inline artifacts in section `text`.
- **BNS chapter markers**: Chapter boundaries are detected from `CHAPTER <roman>` lines in the body text. The BNS index (pages 2–19) provides section headings; the correspondence table (pages 20–73) is skipped.
- **Source-hash safety rule**: If a source PDF hash changes, the corpus for that act is regenerated as an atomic replacement. Records from different PDF versions are never mixed in one corpus file.

## MongoDB collections (Story 2.2)

Four collections in the database named by `MONGODB_DB_NAME` (`MONGODB_TEST_DB_NAME` when `APP_ENV=testing`). Atlas (M0 free tier or higher) is required.

| Collection | Role | Key fields |
|---|---|---|
| `sources` | One record per act PDF | `_id` (`bns_source`/`ipc_source`), `act`, `source_sha256`, `section_count` |
| `sections` | One record per corpus section (858) | `_id` = `section_id`, `act`, `section_number`, `chapter`, `heading`, `text`, `source_id` |
| `chunks` | Overlapping pieces of section text | `_id` = `chunk_id`, `section_id`, `chunk_index`, `text`, `embedding_input`, `act`, `chapter`, `status`, `access_level` |
| `embeddings` | One vector per chunk | `_id` = `embedding_id`, `chunk_id`, `vector` (1,024), `model`, `act`, `chapter`, `status`, `access_level` |

`act`, `chapter`, `status`, `access_level` are copied onto `embeddings` so the vector index can filter without a join. Vectors are never stored on `chunks`.

### Chunking

- `RecursiveCharacterTextSplitter`, `chunk_size=2048`, `chunk_overlap=256`, separators `["\n\n", "\n", ". ", " ", ""]`; chunks never cross sections.
- `embedding_input` = `[{act_label}] {heading}\n{chunk_text}`; `text` keeps the raw chunk.
- `chunk_id = sha256(f"{section_id}:{chunk_index}:{chunking_version}")[:24]`; `CHUNKING_VERSION` (`v1`) is bumped when splitter settings or the `embedding_input` format change.

### Embedding

- `voyage-3.5`, 1,024 dimensions, cosine; documents use `input_type="document"`, queries `"query"`.
- `embedding_id = sha256(f"emb:{chunk_id}:{model}:{model_version}:{dimensions}")[:24]`.
- Batches of 10 chunks, 25 s pause (free-tier limit), up to 4 attempts with growing backoff on Voyage errors. Each batch is written immediately, so an interrupted run resumes. About 35-40 min for ~900 chunks on a free key.

### Vector index

`vector_index` on `embeddings`, type `vectorSearch`, `fields` array: `vector` (`numDimensions` 1024, cosine) plus `filter` fields `act`, `chapter`, `status`, `access_level`. Created with `create_search_index`; the runner waits up to 120 s for `READY`. A different existing index stops the run (drop it manually in Atlas); it is never replaced automatically.

### Ingestion command

```bash
uv run python -m building_with_rag.ingestion.ingest
```

Re-running with unchanged inputs skips everything and makes no Voyage calls. Changed sections or chunks are replaced and re-embedded; removed ones are deleted. The run ends with a sample `$vectorSearch` ("punishment for theft") printing `chunk_id`, `section_id`, heading and score, or `vector query pending` if the index is not ready.

### Limitations

- Atlas required; no `$text`, `$geoNear` or client-side fallback.
- Single `access_level` (`public`).
- Re-ingest is the only update path.
- Dimensions or similarity changes need a manual index drop in Atlas.
- No keyword/hybrid index fields yet.

## Semantic retrieval (Story 2.3)

Code: `src/building_with_rag/retrieval/semantic.py`, routed from `routes/query.py` for `pattern: "semantic"` only.

Flow: validate request → scope filters → embed question → `$vectorSearch` → resolve chunk/section → `QueryResult`.

- Validation: question trimmed (non-empty); `limit` 1–20; `SemanticFilters` forbids unknown fields and accepts only known plain strings (`BNS_2023`/`IPC_1860`, `in_force`/`repealed`, `public`). `caller_id` must be omitted or equal `WEBUI_DEMO_CALLER_ID`; `required_acts` and `chapter` are rejected (422). `generate_answer` is accepted but nothing is generated (noted in `trace.ignored`).
- Scope: server fixes `access_level=["public"]`; caller `act`/`status` lists only narrow it (`$in` inside the `$vectorSearch` `filter`).
- Embedding: raw question, `voyage-3.5`, `input_type="query"`, vector length must be 1,024.
- Search: `vector_index` on `embeddings`, `numCandidates = max(limit, min(200, max(50, 10*limit)))`. Score order kept; chunks (not sections) returned. Hits whose chunk/section can't be resolved are omitted and counted in `trace.unresolved_hits`.
- Outcomes: `ok` (passages), `no_results` (HTTP 200, filters match nothing; no score cutoff), 503 `retrieval_not_ready` (credentials, index, empty embeddings, dimension mismatch), 502 `retrieval_upstream_error` (Voyage/MongoDB failure, e.g. rate limit). Scores rank similarity only; they do not prove correctness.
- New optional `RetrievedChunk` fields: `chunk_index`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `source_pdf`, `source_sha256`, `needs_review`.
- Clients are created lazily with bounded timeouts; `/healthz` works without credentials.

Diagnostic command (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}' \
  | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```
