# Story 5.2 — Query Decomposition and HyDE

Tenth and final course story. Follows Story 5.1. Adds the last two explicitly selectable modes, `decomposition` (`rag-decomposition`) and `hyde` (`rag-hyde`), then smoke-checks all six course modes so later additions have not broken earlier contracts. Both modes are small retrieval extensions on the existing shared path. They are not agents, routers, graph construction, or a new application.

## Purpose

- **Decomposition**: one compound comparison question → one bounded model step → one to three validated subquestions → real corpus evidence retrieved for each → one `QueryResult`. A comparison is never synthesized when any accepted subquestion lacks evidence.
- **HyDE**: one vocabulary-mismatch question → one model-written hypothetical retrieval text → embedded with the existing retrieval model → used only to find real corpus passages. The hypothetical text is never evidence, a citation, or an answer source.

The selected mode decides the route. No automatic routing, no fallback between modes (a failed HyDE never degrades to semantic; a failed decomposition never degrades to a single-question search). Semantic, hybrid, hybrid-reranked and structured stay exactly as they are.

Reuse, do not rename or replace: `QueryRequest`, `QueryResult`, `RetrievedChunk`, `GenerationResult`, the `Pattern`/`rag-<pattern>` registry, `/v1/models`, `QueryError`, `semantic` helpers (`_db`, `_voyage_client`, `_ensure_ready`, `_check_scope`, `effective_filters`, `_vector_filter`, `_chunk_result`), `answer._complete` (non-streamed provider call), the Voyage settings (`voyage-3.5`, 1,024 dimensions, `VOYAGE_API_KEY`) and the generation settings (`GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME`). **Add no environment variables, renamed or provider-specific settings, endpoints, dependencies, UI screens, or `QueryResult` top-level fields.** Do not re-run ingestion or touch indexes. Do not change prompts, constants, validation, confidence rules, or streaming framing in `generation/`.

## Prerequisites

- Stories 1.1, 2.1–2.3, 3.1, 3.2, 4.1, 4.2, 5.1 complete. Read `docs/architecture.md` first.
- `.env` (untracked) with `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`. For the six-mode smoke check also the Story 4.1 keyword index and `RERANK_API_KEY`. Never print or log keys, URLs, or the URI.

### Findings to respect (verified in the repo)

- **`SubquestionEvidence` is not defined anywhere** (Story 1.1 only lists an empty `subquestions` field; `contracts.py` has `subquestions: list[str]`, only ever `[]`). Search once more for an uncommitted definition; if none, add it in `contracts.py` as a new additive model and change only the type of `QueryResult.subquestions` to `list[SubquestionEvidence]` (default empty, so every existing result and placeholder stays valid). Exact fields: `subquestion: str`, `status: Literal["evidenced","insufficient"]`, `results: list[RetrievedChunk]`, `reason: str | None = None`; `extra="forbid"`. Change no other contract.
- `QueryResult.hyde_direct_candidates`, `hyde_query_candidates` (both `list[RetrievedChunk]`) and `hyde_hypothetical_text_debug` (`str | None`) already exist and are unused. Use them as is.
- `semantic._check_scope` rejects `required_acts` and `chapter` and takes the mode name from the request, so both new modes reuse it unchanged (the Pipe already sends `chapter` only with `structured`).
- `run_semantic` and `semantic._search` embed one text per call and Voyage's free tier is rate limited (ingestion pauses 25 s). Decomposition therefore embeds all accepted subquestions in **one** Voyage call; HyDE uses at most two. Do not loop `run_semantic` per subquestion. New shared helpers live in a new module and call existing private helpers without changing their behavior.
- `assemble_context` takes `results` only and caps at 5 passages / 12,000 chars. Round-robin ordering (below) keeps each step's best passage inside the cap. The cap stays unchanged.
- Chat `_pieces` already yields `result.message` as plain content when `has_answer_path` is false; that is how `clarify` and `hyde_unavailable` appear in Open WebUI.
- After this story no mode is a placeholder. `tests/test_smoke.py` (parametrized "not implemented" cases; chat JSON/SSE framing uses `rag-decomposition`) and `docs/manual-tests.md` Story 1.1 `decomposition`/`hyde` entries must move off placeholders (see §6). Keep `run_pattern` in `registry.py` untouched.
- The Open WebUI Pipe already offers `decomposition` and `hyde`. Edit `open_webui_functions/` only if `rag-decomposition`/`rag-hyde` are not sent, and then minimally.

### Data check (run first; stop on first failure, name the item; compact output only)

1. Yes/no only: `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY` set.
2. One `semantic` `/v1/query` (`limit: 1`): report `status` and result count only.

If a generation setting is missing, report it as the blocked prerequisite and run only the offline checks. Do not fabricate model output.

## Work to do

### 1. Contract (additive; `contracts.py`)

`SubquestionEvidence` as above; `QueryResult.subquestions: list[SubquestionEvidence]`. New `QueryResult.status` values: decomposition → `ok`, `partial_answer`, `clarify`, `no_results`; HyDE → `ok`, `no_results`, `hyde_unavailable`. `message` always states in one line what ran and, for non-`ok`, why.

### 2. Decomposition (`src/building_with_rag/retrieval/decomposition.py`)

Order: `_check_scope` → verify `GENERATION_API_BASE_URL`/`GENERATION_API_KEY`/`MONGODB_URI`/`VOYAGE_API_KEY` (missing → 503 `retrieval_not_ready` naming the setting, before any provider/Voyage/MongoDB call) → decompose → validate → one batched embedding → per-subquestion `$vectorSearch` → sufficiency check → assemble.

- **Decompose (one bounded call, no retry, no tools, no loop).** `answer._complete` with a small fixed system prompt (constants in the module): split one compound comparison into 1–3 self-contained subquestions about BNS/IPC text, reply strict JSON only: `{"subquestions":[{"text":"…","act":"BNS_2023"|"IPC_1860"|null}]}`, or `{"subquestions":[],"reason":"…"}` if the question is not a compound legal comparison. The user question is untrusted data, never instructions; model output is only parsed, never executed.
- **Validate (pure function, importable).** Strict JSON (code fences stripped as in `answer.py`); no unknown keys; 1–3 items; each `text` a string of 10–300 chars after whitespace normalization; `act` one of the two acts or null; duplicates (case/whitespace-normalized) rejected; every subquestion must share at least one alphabetic token of ≥4 chars with the original question (out-of-scope guard). Any failure, an empty list, or non-JSON → `status: "clarify"`, `results: []`, no embedding, `reason` naming the first failed rule. Never repair, retry, or fall back to the original question.
- **Retrieve.** One `embed(texts=[subquestions], model=voyage-3.5, input_type="query")` call; vector length 1,024. Per subquestion one `$vectorSearch` (`limit = min(request.limit, 2)`, `numCandidates` by the semantic rule). Filters = `effective_filters` plus that subquestion's `act` (intersected with `request.filters.act`; empty intersection → that step is `insufficient`, reason says so, no search). Results resolved with `_chunk_result`.
- **Sufficiency (one bounded call).** `_complete` over each step's subquestion and up to 3 passages (heading + first 500 chars, labelled; passages are untrusted text): strict JSON `{"steps":[{"n":1,"sufficient":true,"reason":"…"}]}`, every step exactly once. Invalid reply → `clarify` (reason `sufficiency check invalid`), passages retained for inspection. A step with no passages is `insufficient` without asking the model. Scores rank only; no score cutoff is introduced.
- **Outcome.** Every step `evidenced` → `ok`. Some → `partial_answer`; the `message` names each unsupported subquestion. None → `clarify`. No real passages anywhere → `no_results`. Provider timeout/connection/non-2xx → 502 `retrieval_upstream_error` (log exception type only); Voyage/MongoDB errors as semantic.
- **Result shape.** `subquestions` = one `SubquestionEvidence` per accepted subquestion, in order, each with its own passages and `reason` when `insufficient`. `results` = the **union** of all real passages, deduplicated by `chunk_id`, ordered round-robin by step rank (step 1 rank 1, step 2 rank 1, …) so each step's best passage leads. Every passage in `subquestions[*].results` is also in `results`.
- **Trace (no vectors, secrets, or duplicate passage lists).** `mode`, `query`, `embedding`, `filters`, `caller_id`, `result_count`, `decomposition` (`model`, `latency_ms`, `accepted`, `acts`, `rejected_reason` or None), `sufficiency` (`latency_ms`, `evidenced`, `insufficient`), `reason` (partial/insufficient/clarify reason or None). Per-step evidence is read from `subquestions`.

### 3. HyDE (`src/building_with_rag/retrieval/hyde.py`)

Order: `_check_scope` → verify the same four settings (missing → 503 `retrieval_not_ready` naming the setting; never a fabricated hypothetical or a "ran" claim) → direct retrieval → generate → validate → embed → search → merge.

- **Direct candidates.** `run_semantic` on a copy of the request (`pattern=semantic`) → `hyde_direct_candidates` (existing behavior; its errors pass through).
- **Hypothetical text (one call, no retry).** `_complete` with a fixed prompt: write one short passage in the style of a statute section that would answer the question; reply strict JSON `{"text":"…"}`. Validate: JSON object, only `text`, a non-empty string of ≤1,500 chars. Empty, malformed or oversized → `status: "hyde_unavailable"`, `results` and candidate lists empty, direct candidates **not** promoted to results (no pretending HyDE ran). Provider failure → 502 `retrieval_upstream_error`.
- **HyDE candidates.** `embed(texts=[hypothetical], model=voyage-3.5, input_type="document")` (the text imitates a corpus passage; same model and dimensions, so the existing index applies), one `$vectorSearch` with the same effective filters and `limit` → `hyde_query_candidates`.
- **Merge.** `results` = direct and HyDE candidates interleaved (direct first), deduplicated by `chunk_id` (keep the higher score), first `request.limit`. Every result is a real chunk; scores are each route's own similarity and rank only.
- **Debug boundary.** The hypothetical text appears **only** in `hyde_hypothetical_text_debug` on `/v1/query`. Never in `results`, `trace` (length only: `hypothetical_chars`), `message`, generation context, citations, chat output, or logs.
- **Trace.** `mode`, `query`, `embedding` (model, dimensions), `filters`, `caller_id`, `result_count`, `hyde` (`model`, `latency_ms`, `hypothetical_chars`, `direct_count`, `hyde_count`, `overlap`, `final_from` counts: `both`/`direct_only`/`hyde_only`).

### 4. Routing and shared path

- `pipeline.retrieve`: `DECOMPOSITION` → `run_decomposition`; `HYDE` → `run_hyde`. Add both to `REAL_PATTERNS`.
- `has_answer_path`: decomposition → `status in {"ok","partial_answer"}`; hyde → `status == "ok"`; other new statuses return the message only, no model call (query leaves `generation` unset; chat yields `result.message`).
- Decomposition answers use only passages of `evidenced` steps: `answer_events` passes `stream_answer` a shallow copy of the result with `results` = those passages (an `insufficient` step's passages stay in the response for inspection but never reach the context). For `partial_answer`, chat yields `result.message` (names the unsupported part) before the streamed text; `/v1/query` keeps it in `message`. Prompts, validation, confidence, citations, footer labels, `MAX_ATTEMPTS`, and SSE framing are unchanged.
- The hypothetical text can never be cited: it is not a `RetrievedChunk`, so it is outside `assemble_context` by construction.

### 5. Docs

- `docs/architecture.md`: modes paragraph (all six real; `not_implemented` placeholders no longer returned), `QueryResult` (`SubquestionEvidence`, new status values), endpoints/chat statements, context-section "real modes" sentence; add "Query decomposition (Story 5.2)" and "HyDE (Story 5.2)" with flow, validation rules, constants, result shape, trace, outcomes, boundaries, and the diagnostic commands below; add "Course smoke check" with the command in check 7. Limitations: decomposition is one model pass (no iteration, no sub-sub-questions, 1–3 steps); sufficiency is a model judgment, not proof; per-step evidence is bounded (2 passages); the 5-passage/12,000-char answer context still applies; HyDE quality depends on the model's hypothetical and can drift; it adds provider calls (latency, cost, Voyage rate limits), no retry, no fallback; neither mode adds follow-up memory, automatic routing, graph retrieval, or agentic loops; scores rank only; BNS/IPC source status is not current legal applicability.
- `docs/manual-tests.md`: replace the Story 1.1 `decomposition`/`hyde` placeholder query entries and the chat entries (use a `structured` ambiguous question for offline framing); add a compact Story 5.2 section.

### 6. Tests

- `tests/test_decomposition_hyde.py` (offline; fake `answer._complete`, the embed call, and the vector search; few cases).
- `tests/test_smoke.py`: remove the "not implemented" parametrization; assert `/v1/models` lists six IDs and that `QueryRequest` accepts all six patterns; chat JSON/SSE framing cases use `rag-structured` with `What does section 103 say?` (clarification, no credentials, no network).

## Completion checks (compact; never print passages, vectors, keys, hypothetical text, or whole responses)

Diagnostic forms (text truncated):

```bash
# decomposition
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"<q>","pattern":"decomposition","generate_answer":true}' \
  | jq '{status, reason: .trace.reason, t: (.trace | {decomposition, sufficiency}), s: [.subquestions[] | {subquestion, status, reason, ids: [.results[].section_id]}], r: [.results[] | {section_id, score, text: .text[:50]}], g: (.generation | {outcome, confidence, text: .text[:200]})}'

# hyde (hypothetical text length only; never print it)
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"<q>","pattern":"hyde","generate_answer":true}' \
  | jq '{status, t: .trace.hyde, direct: [.hyde_direct_candidates[].section_id], hyde: [.hyde_query_candidates[].section_id], final: [.results[] | {section_id, score}], debug_chars: (.hyde_hypothetical_text_debug | length), g: (.generation | {outcome, confidence, text: .text[:200]})}'
```

1. **Decomposition (prepared BNS-versus-IPC comparison):** use one compound question that clearly names both acts, e.g. "How does the BNS punishment for murder differ from the IPC punishment for murder?" (say why if you pick another). Inspect the diagnostic **before** looking at the chat answer. Expect `status: ok`, 2 accepted subquestions with `evidenced` steps, passages from both acts, every step passage present in `results`, and `generation.citations` drawn from `results` only. Record `status`, each subquestion with section IDs, `trace.decomposition`/`sufficiency` latency, and `generation | {outcome, confidence, text[:200]}`. Read one cited passage (truncated) to confirm it supports its claim.
2. **Decomposition limits (small):** one question with no corpus evidence for one half (e.g. a BNS-vs-IPC comparison that includes a non-criminal-law half) → `partial_answer` with the unsupported subquestion named in `message`, `reason`, and its step; and "What is theft?" (not compound) → `clarify` or one valid subquestion, never an invented comparison. Pause ~25 s between live calls (Voyage free tier).
3. **HyDE (vocabulary mismatch):** one prepared question in lay wording that shares no key statutory term with the section (e.g. "What happens to someone who secretly takes a neighbour's bike?"; say why if you pick another). Inspect before the chat answer. Expect `status: ok`, `hyde_direct_candidates` and `hyde_query_candidates` populated, final `results` deduplicated and each a real corpus chunk, `debug_chars > 0` on `/v1/query` only. Write 3–5 lines on what differed between direct, HyDE, and final (IDs, order, which route found each). Report only what was observed; do not claim HyDE is better in general. Confirm the hypothetical text does not appear in `message`, `trace`, `generation`, or chat.
4. **Missing configuration ≠ result:** with `GENERATION_API_KEY=` empty and the API restarted, both modes return HTTP 503 `retrieval_not_ready` naming the setting (query) and the OpenAI-style error envelope (chat), no results, no Voyage/MongoDB call, while `semantic` still returns `ok`. Restore the key afterwards. Malformed model output is covered offline; do not break live credentials.
5. **Chat / Open WebUI streaming:** `rag-decomposition` and `rag-hyde` via chat (stream, `head -c 1500`) for checks 1 and 3 show DRAFT/confidence/`Sources:` and agree with `/v1/query` on outcome and citations; cited IDs are among `results`. In Open WebUI (`http://127.0.0.1:8080`, `Building with RAG`, RAG-options chip) ask check 1 under `decomposition` and check 3 under `hyde` once each and note what is displayed. If Open WebUI is unavailable, report it as the blocked prerequisite.
6. **Offline tests** (`tests/test_decomposition_hyde.py`; no network; few cases): (a) decomposition validation: valid 2 subquestions accepted; empty, non-JSON, unknown key, 4 items, duplicate, bad `act`, over-length, and no-token-overlap output → `clarify`, no embedding or search call; (b) all steps evidenced → `ok`; one insufficient → `partial_answer` with its reason; none → `clarify`; invalid sufficiency reply → `clarify`; (c) result shape: each step passage is in `results`, no duplicate `chunk_id`, round-robin order, `SubquestionEvidence` rejects unknown fields; (d) answer context from a `partial_answer` contains only `evidenced` step passages; (e) HyDE: valid text → candidate lists, merged deduped `results`, hypothetical only in `hyde_hypothetical_text_debug` (not in `trace`, `message`, `assemble_context` entries); empty/malformed/oversized text → `hyde_unavailable`, empty results, no embedding of it, no promotion of direct candidates; (f) missing generation setting → 503 `retrieval_not_ready` for both modes, nothing called; provider failure → 502; (g) `pipeline.retrieve` routes both modes; `has_answer_path` false for `clarify`/`hyde_unavailable`; scope rejects `required_acts` and `chapter`.
7. **Six-mode smoke (required, final course check):** through `/v1/query`, one prepared question per mode, ~25 s apart, one compact line each; then the same six in Open WebUI.

   ```bash
   q() { curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d "$1" \
     | jq -c '{pattern, status, n: (.results|length), g: (.generation // {} | {outcome, confidence}), err: .detail.code}'; sleep 25; }
   # one call per mode with generate_answer true: semantic, hybrid, hybrid-reranked,
   # structured ("What does BNS section 103 say?"), decomposition (check 1), hyde (check 3)
   ```

   Expect each line: `status: ok`, `n ≥ 1`, `generation.outcome: answered` for answerable questions; `rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked`, `rag-structured` unchanged from Stories 2.3, 4.1, 4.2, 5.1 (fields, statuses, trace keys, footer, `Sources:`); `/v1/models` lists the six IDs. In Open WebUI select each of the six modes once and note, one line per mode, that the answer streamed with DRAFT → confidence → `Sources:` (or the structured message / clarification text). Report any earlier-contract regression by mode and field; fix only regressions caused by this story, and name any other as a finding. If a prerequisite (keyword index, `RERANK_API_KEY`, Open WebUI) is unavailable, report that mode as blocked by name rather than skipping silently.
8. `uv run ruff check src/building_with_rag` passes.

Run only this story's checks (`tests/test_decomposition_hyde.py`, `tests/test_smoke.py`, ruff, the live checks above). Do not run the full suite. Add no evaluation harness, benchmark, regression suite, GraphRAG, agentic loop, automatic routing, production security, or conversation memory. If live prerequisites are unavailable, run only checks 6 and 8 and report the blocked prerequisite by name.

## Handover

Record: files created/changed; data-check results (yes/no, status/count); whether `SubquestionEvidence` already existed; constants used (`MAX_SUBQUESTIONS`, `PER_STEP_LIMIT`, hypothetical length cap); commands run; compact diagnostics for checks 1–3 (decomposition question, subquestions with IDs, step statuses, cited passage checked; HyDE question, direct/HyDE/final IDs and the 3–5-line difference note); check 4 and 5 observations including the Open WebUI result or blocked prerequisite; the six-mode smoke result lines (query and Open WebUI) and any regression found; architecture sections updated; deviations from this story.

Story report (per project instructions): `Completed.`, changed paths, test results — at most 7 lines, ending with the full-suite command (`uv run pytest`) for the developer to run if wanted.
