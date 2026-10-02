"""Hybrid retrieval: Atlas Search keyword route + semantic vector route, fused by RRF (Story 4.1).

Fusion uses ranks only (BM25 and cosine scores are not comparable). The fused score ranks
passages only; it does not prove a passage is correct or answers the question.
"""

import logging

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.retrieval.semantic import (
    QueryError,
    _check_scope,
    _chunk_result,
    _db,
    _not_ready,
    _search,
    effective_filters,
    num_candidates,
)
from building_with_rag.settings import get_settings

log = logging.getLogger(__name__)

RRF_K = 60
ROUTE_WEIGHTS = {"semantic": 1.0, "keyword": 1.0}
KEYWORD_COMMAND = "uv run python -m building_with_rag.ingestion.keyword_index"
FUSION_NOTE = "Fused scores rank only and do not prove correctness."

_ready = False


def route_depth(limit: int) -> int:
    """Per-route candidate depth: at least `limit`, 20..50 otherwise 4x the limit."""
    return max(limit, min(50, max(20, 4 * limit)))


def fuse(semantic_hits: list[dict], keyword_hits: list[dict], limit: int) -> list[dict]:
    """Reciprocal Rank Fusion over two ranked hit lists ({chunk_id, score}, best first).

    Returns the top `limit` entries; `fused_rank` is 1-based over the full fused list.
    """
    fused: dict[str, dict] = {}
    for route, hits in (("semantic", semantic_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            entry = fused.setdefault(hit["chunk_id"], {
                "chunk_id": hit["chunk_id"], "semantic_score": None, "semantic_rank": None,
                "keyword_score": None, "keyword_rank": None, "fused_score": 0.0})
            entry[f"{route}_score"] = float(hit["score"])
            entry[f"{route}_rank"] = rank
            entry["fused_score"] += ROUTE_WEIGHTS[route] / (RRF_K + rank)
    ordered = sorted(fused.values(), key=lambda e: (
        -e["fused_score"],
        e["semantic_rank"] if e["semantic_rank"] is not None else float("inf"),
        e["chunk_id"]))
    for position, entry in enumerate(ordered, start=1):
        entry["fused_rank"] = position
    return ordered[:limit]


def _search_state(coll, name: str) -> bool | None:
    """None if the search index is absent, else whether it is queryable."""
    found = list(coll.list_search_indexes(name))
    if not found:
        return None
    return bool(found[0].get("queryable") or found[0].get("status") == "READY")


def _ensure_ready(settings, db) -> None:
    global _ready
    if _ready:
        return
    if not settings.mongodb_uri or not settings.voyage_api_key:
        raise _not_ready("MONGODB_URI and VOYAGE_API_KEY must be set in .env.")
    vector = _search_state(db[schema.EMBEDDINGS_COLLECTION], schema.VECTOR_INDEX_NAME)
    if vector is None:
        raise _not_ready("vector_index not found; run Story 2.2 ingestion.")
    if not vector:
        raise _not_ready("vector_index is not READY yet; retry shortly.")
    keyword = _search_state(db[schema.CHUNKS_COLLECTION], schema.KEYWORD_INDEX_NAME)
    if keyword is None:
        raise _not_ready(f"{schema.KEYWORD_INDEX_NAME} not found; run: {KEYWORD_COMMAND}")
    if not keyword:
        raise _not_ready(f"{schema.KEYWORD_INDEX_NAME} is not READY yet; retry shortly.")
    sample = db[schema.EMBEDDINGS_COLLECTION].find_one({}, {"model": 1, "dimensions": 1})
    if sample is None:
        raise _not_ready("embeddings collection is empty; run Story 2.2 ingestion.")
    if (sample.get("model") != schema.EMBEDDING_MODEL
            or sample.get("dimensions") != schema.EMBEDDING_DIMENSIONS):
        raise _not_ready("Stored embeddings do not match voyage-3.5/1024; re-run ingestion.")
    _ready = True


def _keyword_search(db, question: str, depth: int, filters: dict) -> list[dict]:
    """Atlas Search `text` operator on chunks.text; the question is only the query value."""
    compound: dict = {"must": [{"text": {"query": question, "path": "text"}}]}
    clauses = [{"in": {"path": path, "value": values}} for path, values in filters.items() if values]
    if clauses:
        compound["filter"] = clauses
    return list(db[schema.CHUNKS_COLLECTION].aggregate([
        {"$search": {"index": schema.KEYWORD_INDEX_NAME, "compound": compound}},
        {"$limit": depth},
        {"$project": {"_id": 0, "chunk_id": 1, "section_id": 1,
                      "score": {"$meta": "searchScore"}}},
    ]))


def _complete(db, hits: list[dict], chunks: dict, sections: dict) -> None:
    """Fetch any chunks/sections the hits need that are not loaded yet."""
    missing = [h["chunk_id"] for h in hits if h["chunk_id"] not in chunks]
    if missing:
        for c in db[schema.CHUNKS_COLLECTION].find({"chunk_id": {"$in": missing}}):
            chunks[c["chunk_id"]] = c
    need = list({c["section_id"] for c in chunks.values()} - set(sections))
    if need:
        for s in db[schema.SECTIONS_COLLECTION].find({"section_id": {"$in": need}}):
            sections[s["section_id"]] = s


def _resolved(hits: list[dict], chunks: dict, sections: dict) -> tuple[list[dict], int]:
    keep = []
    for hit in hits:
        chunk = chunks.get(hit["chunk_id"])
        if chunk is not None and chunk["section_id"] in sections:
            keep.append(hit)
    return keep, len(hits) - len(keep)


def run_hybrid(request: QueryRequest) -> QueryResult:
    global _ready
    settings = get_settings()
    _check_scope(request, settings)
    filters = effective_filters(request)
    depth = route_depth(request.limit)
    candidates = num_candidates(depth)
    try:
        db = _db(settings) if settings.mongodb_uri else None
        if db is None or not settings.voyage_api_key:
            raise _not_ready("MONGODB_URI and VOYAGE_API_KEY must be set in .env.")
        _ensure_ready(settings, db)
        sem_raw, chunks, sections = _search(
            settings, db, request.question, depth, candidates, filters)
        kw_raw = _keyword_search(db, request.question, depth, filters)
        _complete(db, kw_raw, chunks, sections)
    except QueryError:
        _ready = False
        raise
    except Exception as exc:  # noqa: BLE001 - any Mongo/Voyage failure; never leak URIs or keys
        _ready = False
        log.warning("hybrid retrieval upstream error: %s", type(exc).__name__)
        raise QueryError(502, "retrieval_upstream_error",
                         "Voyage or MongoDB request failed; retry shortly.") from None

    sem_hits, sem_unresolved = _resolved(sem_raw, chunks, sections)
    kw_hits, kw_unresolved = _resolved(kw_raw, chunks, sections)
    unresolved = sem_unresolved + kw_unresolved
    if (sem_raw or kw_raw) and not (sem_hits or kw_hits):
        _ready = False
        raise _not_ready("Hits could not be resolved to chunks/sections; re-run ingestion.")

    results: list[RetrievedChunk] = []
    for entry in fuse(sem_hits, kw_hits, request.limit):
        chunk = chunks[entry["chunk_id"]]
        result = _chunk_result(chunk, sections[chunk["section_id"]], entry["fused_score"])
        for name in ("semantic_score", "semantic_rank", "keyword_score", "keyword_rank",
                     "fused_score", "fused_rank"):
            setattr(result, name, entry[name])
        results.append(result)

    both = sum(1 for r in results if r.semantic_rank is not None and r.keyword_rank is not None)
    semantic_only = sum(1 for r in results if r.keyword_rank is None)
    keyword_only = sum(1 for r in results if r.semantic_rank is None)
    message = (f"Returned {len(results)} passage(s) by fused rank. " if results
               else "No passages match the filters. ") + FUSION_NOTE
    return QueryResult(
        pattern=request.pattern.value,
        status="ok" if results else "no_results",
        message=message,
        trace={
            "mode": "hybrid",
            "query": request.question,
            "embedding": {"model": schema.EMBEDDING_MODEL, "input_type": "query",
                          "dimensions": schema.EMBEDDING_DIMENSIONS},
            "filters": filters,
            "caller_id": settings.webui_demo_caller_id,
            "result_count": len(results),
            "unresolved_hits": unresolved,
            "semantic": {"index": schema.VECTOR_INDEX_NAME, "limit": depth,
                         "num_candidates": candidates, "hit_count": len(sem_hits)},
            "keyword": {"index": schema.KEYWORD_INDEX_NAME, "path": "text", "operator": "text",
                        "limit": depth, "hit_count": len(kw_hits)},
            "fusion": {"method": "rrf", "k": RRF_K, "weights": dict(ROUTE_WEIGHTS),
                       "route_depth": depth},
            "contribution": {"both": both, "semantic_only": semantic_only,
                             "keyword_only": keyword_only},
        },
        results=results,
    )
