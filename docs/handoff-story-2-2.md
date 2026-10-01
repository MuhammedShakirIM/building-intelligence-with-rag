# Handoff — Story 2.2 (MongoDB vector index)

Branch: `story/story-2-2-mongodb-vector-index`. All changes uncommitted (you manage git).

## Done
- Reviewed old story + implementation. Found: index def used legacy `knnVector`/`mappings` under `type: "vectorSearch"` (Atlas expects `fields` array: `vector` + `filter` entries, `numDimensions`). Story itself had this flaw plus typos (`vectr`, `similarty`, `model_verson`).
- Patched handout: `/Users/shakir/learning/courses/building-intelligence-with-rag/handouts-session-2/handout-2-2-mongodb-vector-index.md`
  (Atlas M0 assumed, vectorSearch `fields` format, filter-field scope, pinned chunk size/overlap, Voyage rate-limit/resume guidance, richer sample-query output).
- Regenerated story (hand-written from patched handout, not via fresh assistant): `docs/stories/story-2-2-mongodb-vector-index.md`.
- Probe: Atlas reachable (MongoDB 8.0.34), `MONGODB_URI` and `VOYAGE_API_KEY` set, ruff + imports OK, corpus = 358 + 500 = 858 sections, ~923 chunks, embedding ~38 min (10 chunks/batch, 25s delay).

## Not done (waiting for your green light)
- No code changed. `ingestion/ingest.py` and `ingestion/mongodb_schema.py` still use the legacy index and do NOT match the new story.
- Ingestion never run. Index format not yet tested on Atlas.

## Known gaps in current code (fix on dry run)
1. Index definition + compat check -> `vectorSearch` `fields` format (`numDimensions`, filters: act, chapter, status, access_level).
2. Sample query catches `PyMongoError` around Voyage call -> should catch `VoyageError`; should print chunk_id, section_id, heading, score (+ `filter` access_level, `numCandidates` 50).
3. Prereq check crashes if `__prereq_check_temp__` leftover; story now says use `list_search_indexes()` instead of temp collection.
4. No 429 backoff beyond one 30s retry; embedding orphan cleanup only runs when something was embedded; removed-section cleanup and `embedding_input` change detection missing (new story requires them).
5. `docs/architecture.md` section describes the old `knnVector` index -> rewrite.

## Resume at home
1. Say "green light dry run" -> I patch code to match story, then run:
   `cd /Users/shakir/learning-capstones/building-intelligence-with-rag && uv run python -m building_with_rag.ingestion.ingest`
   (~40 min; resumable. Use `APP_ENV=testing` to target the test DB first.)
2. Re-run once: everything must skip, no Voyage calls.
3. Lint: `uv run ruff check src/building_with_rag/ingestion`
4. Optional: run patched handout in a clean assistant session to see what participants' stories look like.
5. Make sure `.env` exists on the home machine (untracked) with Atlas URI + Voyage key; Atlas IP access list must allow the home IP.
