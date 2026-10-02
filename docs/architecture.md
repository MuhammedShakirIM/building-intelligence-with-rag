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

`semantic` is real on `POST /v1/query` (Stories 2.3, 3.1) and on chat `rag-semantic` (Story 3.2); `hybrid` is real on both (Story 4.1); `hybrid-reranked` is real on both (Story 4.2). Every other mode returns an honest `not_implemented` placeholder on both until its own story adds behavior. The selected mode decides the route; there is no automatic routing or fallback between modes.

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
- Additive empty-by-default fields later modes use: `omitted_candidates` (used by `hybrid-reranked` since Story 4.2: every candidate cut before or after re-ranking), `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`.
- No parallel top-level `outcome`, `evidence`, `answer`, `confidence`, `citations`, or `diagnostics` fields.

### `RetrievedChunk`

A retrieved passage is always this shape: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and available source fields. Later stories add only the existing hybrid/rerank fields: hybrid route evidence (Story 4.1) and, optional and default `None`, `rerank_score`, `rerank_rank`, `omitted_reason` (Story 4.2).

### Endpoints

- `GET /healthz` — safe, no credentials required.
- `POST /v1/query` — accepts `QueryRequest`, returns `QueryResult`; `semantic`, `hybrid` and `hybrid-reranked` run real retrieval, other modes return placeholders via the shared `run_pattern` path.
- `GET /v1/models` — lists the six `rag-<pattern>` model IDs.
- `POST /v1/chat/completions` — OpenAI-compatible, text-only `ChatCompletionRequest`: `model`, `messages` with `system`/`developer`/`user`/`assistant` roles, `stream`, `n`, and optional strict `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`).

The chat adapter maps the selected model to the same `QueryRequest` and the shared `pipeline.retrieve` + `answer_events` path as `/v1/query`; it sets server-side demo `caller_id` and `generate_answer`. Supports normal OpenAI Chat Completions JSON responses and role/content/stop frames, plus the OpenAI-style error envelope before streaming begins. `rag_options` accepts flat `act`/`status` and the Pipe's nested `filters.{act,status}`. When `CAPSTONE_API_KEY` is non-empty, chat requires `Authorization: Bearer <key>` (constant-time compare, else 401 `invalid_api_key`); other endpoints are unchanged. No custom SSE events.

## Open WebUI (trainer-supplied, separate client)

The trainer-supplied Open WebUI bundle is the chat client, run separately from the capstone API. Its pre-provisioned Pipe sends the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` to the capstone's `/v1/chat/completions`. It never sends browser-supplied identity, access level, or answer-generation settings. The capstone's adapter accepts that exact request and uses server-side `caller_id`/`generate_answer`.

`/v1/query` owns the `QueryResult` diagnostics; Open WebUI receives only readable text from the same generation operation: DRAFT-prefixed streamed answer, retry notice, and a confidence/sources footer (Story 3.2). The full `GenerationResult` stays in `QueryResult.generation`.

## Environment

- `.env` is untracked; secrets are never committed.
- The application must start with no database or model credentials and expose a safe `GET /healthz`.
- Canonical environment values are defined in `.env.example`.
- `RERANK_API_KEY` (empty by default) is required only by `hybrid-reranked`; see "Re-ranking (Story 4.2)".

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
- Keyword search uses the Atlas Search index `chunk_text_index` on `chunks` (Story 4.1); `ingest` ignores it.

## Semantic retrieval (Story 2.3)

Code: `src/building_with_rag/retrieval/semantic.py`, routed from `routes/query.py` for `pattern: "semantic"` only.

Flow: validate request → scope filters → embed question → `$vectorSearch` → resolve chunk/section → `QueryResult`.

- Validation: question trimmed (non-empty); `limit` 1–20; `SemanticFilters` forbids unknown fields and accepts only known plain strings (`BNS_2023`/`IPC_1860`, `in_force`/`repealed`, `public`). `caller_id` must be omitted or equal `WEBUI_DEMO_CALLER_ID`; `required_acts` and `chapter` are rejected (422). `generate_answer` triggers Story 3.1 answer generation (see below).
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

## Hybrid retrieval (Story 4.1)

Code: `src/building_with_rag/retrieval/hybrid.py`, routed by `pipeline.retrieve` for `pattern: "hybrid"` (model `rag-hybrid`). Same corpus, database, chunks and embeddings as semantic; the selected mode decides the route (no automatic routing, no fallback to semantic).

- Keyword route: Atlas Search `$search` with the `text` operator (BM25, `{$meta: "searchScore"}`) on `chunks.text`, index `chunk_text_index` (type `search`). Not `$text`, not the `vectorSearch` index. The question is only the `text.query` value.
- Index definition (`mongodb_schema.KEYWORD_INDEX_DEFINITION`): `dynamic: false`; `text` string with `lucene.standard`; `act`, `status`, `access_level` as `token`. Filters use `in` inside `compound.filter` with the same effective filters as semantic (server-fixed `access_level=["public"]`).
- Create/reuse: `uv run python -m building_with_rag.ingestion.keyword_index` (idempotent, waits up to 120 s for READY, no Voyage calls, no data writes; a different existing index is reported and never replaced). `ingest` ignores this index.
- Fusion: Reciprocal Rank Fusion over ranks only. Each route returns its top `ROUTE_DEPTH = max(limit, min(50, max(20, 4*limit)))` chunks (semantic keeps the Story 2.3 `numCandidates` rule). `fused_score = Σ 1/(RRF_K + rank)`, `RRF_K = 60`, equal weights. Order: `fused_score` desc, then `semantic_rank` (missing last), then `chunk_id`. Chunks are fused by `chunk_id`; top `limit` returned.
- New optional `RetrievedChunk` fields (default `None`; semantic leaves all `None`): `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`. In hybrid `score == fused_score`; a `None` route did not return the chunk.
- Trace: `mode`, `query`, `embedding`, `filters`, `caller_id`, `result_count`, `unresolved_hits`, `semantic`, `keyword`, `fusion` (`method`, `k`, `weights`, `route_depth`), `contribution` (`both`, `semantic_only`, `keyword_only`).
- Outcomes: `ok`, `no_results` (both routes empty), 503 `retrieval_not_ready` (credentials, either index missing/not ready, empty embeddings; the message names the keyword-index command), 502 `retrieval_upstream_error`. A missing keyword index never degrades silently to semantic-only.
- Shared path: context assembly, generation, validation, confidence and chat footer are the same as semantic (`pipeline.REAL_PATTERNS`).
- Limitations: rank-only fusion ignores score magnitude; the `text` operator matches any query term (OR), so long natural-language questions can pull in common words; section numbers match only when they appear in chunk `text`; no stemming/synonyms beyond the standard analyzer. The fused score ranks only and does not prove correctness.

Diagnostic command (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"criminal breach of trust","pattern":"hybrid","limit":5}' \
  | jq '{status, t: (.trace | {semantic, keyword, fusion, contribution}), r: [.results[] | {section_id, score, sr: .semantic_rank, kr: .keyword_rank, fr: .fused_rank, text: .text[:60]}]}'
```

## Re-ranking (Story 4.2)

Code: `src/building_with_rag/retrieval/rerank.py`, routed by `pipeline.retrieve` for `pattern: "hybrid-reranked"` (model `rag-hybrid-reranked`). Flow: hybrid retrieval → bounded candidates → one provider re-rank call → final evidence, then the shared context/generation path. No automatic mode choice, no fallback to hybrid.

- Settings (existing names): `RERANK_API_BASE_URL` (`https://api.voyageai.com/v1`), `RERANK_API_KEY` (empty), `RERANK_MODEL_NAME` (`rerank-2.5`), `RERANK_REQUEST_TIMEOUT_SECONDS` (30), `RERANK_CANDIDATE_LIMIT` (20), `RERANK_SEND_LIMIT` (10), `RERANK_RETURN_LIMIT` (5). Valid when `1 ≤ RETURN ≤ SEND ≤ CANDIDATE ≤ 20` and timeout ≥ 1; otherwise 503 `retrieval_not_ready` naming the setting.
- Request: one `POST {RERANK_API_BASE_URL}/rerank` via `httpx`, Bearer key, `{model, query, documents}` (no `top_k`), no retries. Document = `"{heading}\n{chunk text}"`; the question is only `query`.
- Reply validation: `data` non-empty list; each `index` an int within the sent range, unique; each `relevance_score` finite; every sent candidate scored. Anything else → 502. No score is invented or defaulted.
- Selection (`rerank.select`, pure): candidates = hybrid top `RERANK_CANDIDATE_LIMIT` (`fused_rank` = pre-rerank order); sent = first `RERANK_SEND_LIMIT`, the rest cut before re-ranking (`omitted_reason: not_sent_to_reranker`, rerank fields `None`); sent ordered by `relevance_score` desc, ties by `fused_rank`, `rerank_rank` 1-based; final = first `min(request.limit, RERANK_RETURN_LIMIT)`, others cut after re-ranking (`below_return_limit`, rerank fields kept).
- Result shape: `results` = final evidence by `rerank_rank` with `score == rerank_score` (non-increasing) and all hybrid fields kept; `omitted_candidates` = all cut candidates in `fused_rank` order. Before/after order and scores are read from the records.
- Trace: `mode`, `query`, `filters`, `caller_id`, `result_count`, `hybrid` (`embedding`, `semantic`, `keyword`, `fusion`, `contribution`, `unresolved_hits`), `rerank` (`model`, limits, counts `candidates`/`sent`/`returned`/`omitted_before`/`omitted_after`, `latency_ms`, `usage_tokens` when reported). No vectors or secrets.
- Outcomes: `ok`; `no_results` (no candidates, no provider call); missing/invalid configuration incl. empty `RERANK_API_KEY` → 503 `retrieval_not_ready` before any Voyage/MongoDB call; timeout, connection error, non-2xx or invalid reply → 502 `retrieval_upstream_error`; hybrid errors pass through. Never returns hybrid order labelled as re-ranked.
- Limitations: candidates outside the top `RERANK_CANDIDATE_LIMIT` are never seen; each passage is scored independently against the question; scores are model-specific, uncalibrated, not comparable with fused scores or across questions, and there is no score cutoff; one extra provider call per request (latency, cost, rate limits); no retry, no fallback; the answer context cap (5 passages / 12,000 chars) still applies. Scores rank only and do not prove correctness.

Diagnostic command (text truncated; merges returned and omitted records):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"<q>","pattern":"hybrid-reranked","limit":5}' \
  | jq '{status, t: .trace.rerank, r: [(.results[]|.+{kept:true}), (.omitted_candidates[]|.+{kept:false})] | map({section_id, kept, fr: .fused_rank, fs: .fused_score, rr: .rerank_rank, rs: .rerank_score, why: .omitted_reason, text: .text[:50]}) | sort_by(.fr)}'
```

## Context and answer boundaries (Story 3.1)

Code: `src/building_with_rag/generation/{context,answer}.py`, called from `routes/query.py` only for real modes (`semantic`, `hybrid`, `hybrid-reranked`) with `generate_answer: true`.

Flow: semantic `QueryResult` → bounded labelled context (`E1`…; at most 5 passages / 12,000 chars, never cut mid-passage) → one `POST {GENERATION_API_BASE_URL}/chat/completions` (`temperature: 0`, 30 s timeout, no retries) → strict JSON parse → citations resolved from the supplied context only.

- Outcomes (`generation.outcome`): `answered` (non-empty `text`, claims each citing supplied labels, resolved `citations`/`supporting_passages`), `insufficient_evidence` (also when context is empty; no model call), `unavailable` (missing settings, timeout, connection error, non-2xx), `malformed` (non-JSON, missing `choices`, unknown label, rule violation; no repair or retry).
- Added optional `GenerationResult` fields: `outcome`, `claims`, `citations`, `supporting_passages`, `provider`, `trace`, `context_outcome` (existing `text`, `model` kept).
- Evidence blocks are untrusted source text, never instructions. No claims of current legal applicability beyond the supplied BNS/IPC documents.
- `unavailable` and `malformed` return HTTP 200 with retrieval `results` intact; `status` stays the retrieval status. Prompts, URL and key never appear in responses.
- Story 3.2 supersedes the single non-streamed JSON call below: see next section.

## Streamed answers and confidence (Story 3.2)

- Shared path: `pipeline.retrieve` → `pipeline.answer_events` (`generation.answer.stream_answer`). `/v1/query` drains the events into `QueryResult.generation`; chat forwards `text`/`notice` events and renders the footer from the same final result. One request = one generation/validation operation.
- Event flow: retrieval/context (non-streamed) → attempt: provider `stream: true` SSE text with inline `[E1]` labels (`INSUFFICIENT_EVIDENCE: <reason>` sentinel is buffered, never streamed) → checks → pass, or notice and retry. `MAX_ATTEMPTS = 2`; 30 s timeout per call.
- Checks per attempt, in order: `citation_labels` (unknown labels), `claim_cited` (every sentence/bullet cited), `support` (one non-streamed validator call, strict JSON). Citations are never stripped or rewritten.
- New optional `GenerationResult` fields: `confidence` (`high` only when the final attempt passes; `low` when it fails a check; `None` when nothing could be judged), `issues` (`attempt`, `check`, `detail`; all attempts), `attempts` (`attempt`, `status` passed|failed|unjudged, `chars`, `latency_ms`), `draft_answer` (last unpassed text), `low_confidence_reason`.
- Outcome mapping: passed → `answered` (high); failed final check → `malformed`, empty `text`, `draft_answer`, `issues`, `low`; provider failure → `unavailable`; validator unreachable → `unavailable` + `draft_answer`; validator invalid reply → `malformed` + `draft_answer`; `insufficient_evidence` and empty context unchanged.
- Chat labels: each attempt starts `DRAFT — checking evidence`; failed non-final attempt → `Check failed: …. Retrying (attempt 2 of 2)…`; pass → `Evidence check passed — confidence: high` + `Sources:` lines; final failure → `DRAFT — low confidence, not the final answer.` + reason/details; insufficient evidence → one plain sentence.
- Failure after text began: one line `Answer generation unavailable — the text above is an unchecked draft.`, then `stop` and `[DONE]` (HTTP 200). Errors before streaming (auth, model, retrieval) are HTTP errors.
