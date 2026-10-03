# Story 6.1 — Bounded Agentic RAG

Eleventh and final additive course extension. Follows Story 5.2. Adds the seventh selectable mode, `agentic` (`rag-agentic`). It is deterministic application code: a small read-only controller over three existing adapters. It is not an LLM planner, router, loop, or new application.

## Purpose

Explicitly selected `agentic` runs a fixed controller: validate → choose one of three fixed paths → call at most two allowlisted adapters → inspect every result → stop with a traced reason. Real retrieved corpus passages stay the only evidence, answer and citation source.

Selecting `agentic` is the only trigger. Its internal path choice never routes requests from any other mode, and no other mode falls back to it.

Reuse, do not rename or replace: `Pattern`/`PATTERN_MODEL_IDS`, `QueryRequest`, `QueryResult`, nested `GenerationResult`, `RetrievedChunk`, `QueryError`, `structured.classify`, `run_structured`, `run_semantic`, `run_hyde`, `semantic.effective_filters`, `pipeline.retrieve`/`answer_events`/`attach_generation`, `POST /v1/query`, `GET /v1/models`, `POST /v1/chat/completions`, Open WebUI streaming, all provider settings. **Add no** provider variables, endpoints, dependencies, UI screens, `QueryResult` top-level fields, or contract models. Do not change `generation/`, existing adapters, `registry.run_pattern`, or `open_webui_functions/` (the Pipe already offers `agentic` and sends `rag-agentic`; its `graph` option stays an unknown model).

## Prerequisites

- Stories 1.1–5.2 complete. Read `docs/architecture.md` first.
- `.env` (untracked) with `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`; for the seven-mode smoke also the Story 4.1 keyword index and `RERANK_API_KEY`. Never print keys, URLs, or the URI.
- Data check (compact, stop on first failure): yes/no for the four settings; one `semantic` `/v1/query` (`limit: 1`) reporting `status` and result count only. If a setting is missing, report it as the blocked prerequisite and run only offline checks.

## Work to do

### 1. Registry and routing

- `registry.py`: add `Pattern.AGENTIC = "agentic"` and `"rag-agentic"` to `PATTERN_MODEL_IDS` (appended last). No other entry changes. `/v1/models` then lists seven IDs.
- `pipeline.py`: `retrieve` routes `AGENTIC` → `run_agentic`; add `agentic` to `REAL_PATTERNS`; `has_answer_path` for `agentic` is true only for `status == "ready_for_synthesis"`. Update the "six" docstrings (`pipeline.py`, `routes/models.py`).
- `routes/query.py` and `routes/chat.py` stay unchanged: non-answer statuses already return `message` only (chat yields `result.message`); answers use the existing grounded-answer, confidence, citation, diagnostic and streamed paths.

### 2. Setting (`settings.py`, `.env.example`)

`AGENTIC_DEADLINE_SECONDS`, default `20`, valid 1–120 (not a provider variable). Invalid → 503 `retrieval_not_ready` naming it, before any tool call.

### 3. Controller (`src/building_with_rag/retrieval/agentic.py`, new)

**Allowlist (constant, fixed, not extensible at runtime):** `structuredLookup` → `run_structured`, `semanticRetrieve` → `run_semantic`, `hydeRetrieve` → `run_hyde`. Each call uses `request.model_copy(update={"pattern": <adapter pattern>})`, so `caller_id`, `filters`, `limit` and the server-fixed `access_level=["public"]` propagate unchanged. No other tool, raw database, shell, HTTP, write, browse, memory, retry, parallel, background or recursive work.

**Scope check (before anything else):** invalid `caller_id` → 422 `invalid_caller_id`; `required_acts` or `chapter` set → 422 `unsupported_option` naming agentic mode (same as `semantic._check_scope`).

**Validation (pure, offline, before any tool call).** Use `structured.classify` for section detection only. Trace records a fixed `reason_code`, never the question.

| Input | Result |
|---|---|
| bare section number, act missing, both acts, several sections, number outside 1–999 or non-integer | `clarify` (`bare_section_number`, `act_missing`, `both_acts`, `several_sections`, `section_out_of_range`) |
| counting/listing phrasing (`classify` aggregation/filter) | `clarify` (`listing_or_count`) |
| `act` filter excludes the named act | `clarify` (`act_filter_conflict`) |
| fewer than 3 alphabetic words and not an act-qualified section request | `clarify` (`too_short`) |
| anything else | valid |

**Fixed rules, in order:**

1. **Act-qualified section request** (`classify` status `ok`): call `structuredLookup` once. Adequate only if status `ok` with exactly one usable passage whose `act` and `section_number` equal the classified values → `ready_for_synthesis`. `not_found` → `partial_result` (`record_not_found`, empty results). Adapter `clarification_needed` → `clarify`. Never guess a section, never fall back to semantic/hyde.
2. **Vocabulary-mismatch rule (prepared, fixed table).** Applies when the question contains a whole-word term from the constant `LAY_TERMS` tuple (starter: `bike`, `scooter`, `swindle`, `scam`, `beat up`, `break into`, `mobile phone`). Then call `semanticRetrieve`; the direct result is adequate only if at least one usable passage's heading or text contains the matched term (statute wording rarely does — that is the mismatch). If not adequate, call `hydeRetrieve` once, after it; adequate = at least one usable passage. Retained evidence = the last adequate call's `results` only.
3. **Ordinary valid question:** call `semanticRetrieve` once.

**Adequacy rule (one documented rule):** a *usable passage* is a `RetrievedChunk` with non-empty `chunk_id`, `section_id` and non-blank `text`, `act` in `BNS_2023`/`IPC_1860`. Adequate = at least one usable passage (rule 2's direct step adds the lay-term check above). A returned passage outside the effective `act`/`status` filters, or more results than `limit`, makes the result **invalid** (`tool_result_invalid`). Controller never re-ranks, edits, merges or dedupes adapter results.

**Outcomes (`QueryResult.status`, new additive values `ready_for_synthesis`, `partial_result`; `clarify` already exists):**

- `ready_for_synthesis`: adequate evidence; `results` = those real passages; answer path runs.
- `partial_result`: valid request, no adequate evidence or a stop condition. `results` = usable real passages already returned (inspection only, may be empty); no model call.
- `clarify`: validation failure or adapter clarification; empty results; no model call.

**Stop conditions (all explicit in trace, none retried):** `tool_unavailable` (adapter `QueryError` or status `hyde_unavailable`; record HTTP code only), `tool_result_invalid`, `deadline_expired`, `budget_exhausted`. Budget: at most 2 adapter calls, at most 1 `hydeRetrieve`, enforced by a guard before every call. Deadline: monotonic clock checked before each call (no call if expired) and after (a result returned past the deadline is discarded and not inspected). The deadline is cooperative: one in-flight adapter call keeps its own existing timeout, so wall time may exceed the deadline by one call. Adapter failures become `partial_result` (HTTP 200) with the trace reason, not 502/503.

**Boundaries:**
- The HyDE hypothetical text never enters the controller. Copy only `results` from adapter output; leave `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug` empty/`None`. Never copy an adapter `trace` (it holds `query`).
- `message` is a fixed one-line string (what ran and why it stopped); never echoes the question.
- No log of question text, passages, or secrets; log adapter exception type only.

**Trace (`trace`, no raw question, credentials, vectors, passages or hypothetical):**
`mode: "agentic"`, `caller_id`, `filters`, `limit`, `validation` (`outcome` valid|clarify, `reason_code`), `path` (`clarify`|`exact_lookup`|`semantic`|`vocab_mismatch`), `steps` (per call: `n`, `adapter`, `tool_status`, `returned`, `usable`, `adequate`, `ms`), `call_count`, `hyde_calls`, `deadline_seconds`, `elapsed_ms`, `stop_reason` (`evidence_adequate`|`no_usable_evidence`|`record_not_found`|`needs_clarification`|`tool_unavailable`|`tool_result_invalid`|`deadline_expired`|`budget_exhausted`), `result_count`.

### 4. Tests

Only update existing assertions in `tests/test_smoke.py` that fix the count at six (`/v1/models` lists seven, `QueryRequest` accepts all seven). Add no new test files, harness, or regression suite.

### 5. Docs

- `docs/architecture.md`: modes table (seven), "six" statements, `QueryResult` statuses, endpoints/chat text, context-section "real modes" sentence; add "Agentic RAG (Story 6.1)" (allowlist, validation table, fixed rules, adequacy, outcomes, stop reasons, trace, boundaries, diagnostic command); update "Course smoke check" to seven modes. Limitations: fixed deterministic controller, not a planner or loop; at most 2 calls/1 HyDE; the `hydeRetrieve` adapter performs its own direct semantic search inside its one call; `LAY_TERMS` is a small fixed table and misses other lay wording; adequacy means "a real passage exists", not "answers the question"; deadline is cooperative; no memory, automatic routing, GraphRAG, writes, retries; BNS/IPC status is not current legal applicability. Update the Story 5.2 limitation sentence about "agentic loops" to point here.
- `docs/manual-tests.md`: compact Story 6.1 section with the commands below. Add `AGENTIC_DEADLINE_SECONDS` to the environment notes.

## Completion checks (compact; never print passages, vectors, keys, hypothetical text or whole responses)

Diagnostic form (text truncated; `trace` holds no passage text):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"<q>","pattern":"agentic","generate_answer":true}' \
  | jq '{status, t: .trace, r: [.results[] | {section_id, score}], dbg: (.hyde_hypothetical_text_debug), hc: (.hyde_query_candidates|length), g: (.generation // {} | {outcome, confidence, c: (.citations|length), text: .text[:150]})}'
```

1. **Act-qualified lookup:** "What does BNS section 103 say?" Expect `path: exact_lookup`, one `structuredLookup` step, `call_count: 1`, `ready_for_synthesis`, `evidence_adequate`, result `bns:103` only. Also "What does IPC section 4 say?" (absent from corpus, no `generate_answer`) → `partial_result`, `record_not_found`, empty `results`, no guess.
2. **Ordinary explanation:** "What is the punishment for theft?" Expect `path: semantic`, one `semanticRetrieve` step, `adequate: true`, `ready_for_synthesis`, `generation.outcome: answered`, every citation among `results`.
3. **Vocabulary mismatch:** "What happens to someone who secretly takes a neighbour's bike?" Expect `path: vocab_mismatch`, `semanticRetrieve` first; record whether the direct step was adequate. If not, exactly one `hydeRetrieve` follows (`call_count: 2`, `hyde_calls: 1`); if adequate, no HyDE call and the story says so. Either way: `results` are real corpus chunks, `hyde_hypothetical_text_debug` is `null`, candidate lists empty, and the trace/message contain no hypothetical text. If the prepared question never reaches HyDE, try one other `LAY_TERMS` question and say why.
4. **Ambiguous input:** "section 103" → `clarify`, `validation.reason_code: bare_section_number`, `call_count: 0`, empty `steps` and `results`, no `query` key anywhere in `trace`, no model call.
5. **Open WebUI stream (one answerable case, check 2):** `curl -s` the chat endpoint with `rag-agentic`, `stream: true`, piped to `head -c 1500`. Expect DRAFT text, `Evidence check passed — confidence: high` and `Sources:` lines, agreeing with check 2 on outcome and citations. If Open WebUI (`http://127.0.0.1:8080`, `Building with RAG`) is available, ask it once under `agentic` and note what is displayed; otherwise report it blocked.
6. **Seven-mode smoke (~25 s apart, one compact line each):**

   ```bash
   q() { curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d "$1" \
     | jq -c '{pattern, status, n: (.results|length), g: (.generation // {} | {outcome, confidence}), err: .detail.code}'; sleep 25; }
   c() { curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
     -d "{\"model\":\"rag-$1\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}]}" \
     | jq -c '{model, len: (.choices[0].message.content|length), sources: (.choices[0].message.content|contains("Sources:")), err: .detail.error.code}'; sleep 25; }
   # q and c once per mode: semantic, hybrid, hybrid-reranked, structured, decomposition, hyde, agentic
   ```

   Expect `/v1/models` lists seven IDs; earlier six modes' statuses, fields and footer unchanged from Stories 2.3–5.2 (`ok`/`answered`; structured and decomposition as before); agentic `ready_for_synthesis` with `sources: true`. Name any mode blocked by a missing prerequisite (keyword index, `RERANK_API_KEY`); report regressions by mode and field, fix only those this story caused.
7. `uv run ruff check src/building_with_rag` and `uv run pytest tests/test_smoke.py` pass.

Run only this story's checks. Do not run the full suite. If live prerequisites are unavailable, run only check 7 plus the offline-capable check 4 and report the blocker by name.

## Handover

Record: files created/changed; data-check results (yes/no, status/count); `LAY_TERMS` and `AGENTIC_DEADLINE_SECONDS` values used; compact trace summaries for checks 1–4 (path, steps, call_count, stop_reason, status); for check 3 whether HyDE ran and why; check 5 result or blocked prerequisite; the seven-mode `q`/`c` result lines and any regression; architecture sections updated; deviations.

Story report (per project instructions): `Completed.`, changed paths, test results — at most 7 lines, ending with the full-suite command (`uv run pytest`).
