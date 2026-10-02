"""Semantic retrieval: embed the question, run an Atlas $vectorSearch, return source passages.

Scores rank similarity only; they are not proof that a passage is correct or answers the question.
"""

import logging
import threading

from pymongo import MongoClient
from voyageai import Client as VoyageClient

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import get_settings

log = logging.getLogger(__name__)

MIN_CANDIDATES = 50
MAX_CANDIDATES = 200
CANDIDATE_MULTIPLIER = 10
MONGO_TIMEOUT_MS = 5000
MONGO_SOCKET_TIMEOUT_MS = 15000
VOYAGE_TIMEOUT_S = 15.0
SERVER_ACCESS_LEVELS = [schema.DEFAULT_ACCESS_LEVEL]
SIMILARITY_NOTE = "Scores rank similarity only and do not prove correctness."

_lock = threading.Lock()
_mongo: MongoClient | None = None
_voyage: VoyageClient | None = None
_ready = False


class QueryError(Exception):
    """Request or retrieval problem mapped to an HTTP status by the route."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def num_candidates(limit: int) -> int:
    """At least `limit` and 50, at most 200; otherwise 10x the limit."""
    return max(limit, min(MAX_CANDIDATES, max(MIN_CANDIDATES, limit * CANDIDATE_MULTIPLIER)))


def _db(settings):
    global _mongo
    with _lock:
        if _mongo is None:
            _mongo = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
                connectTimeoutMS=MONGO_TIMEOUT_MS,
                socketTimeoutMS=MONGO_SOCKET_TIMEOUT_MS,
            )
    name = settings.mongodb_test_db_name if settings.app_env == "testing" else settings.mongodb_db_name
    return _mongo[name]


def _voyage_client(settings) -> VoyageClient:
    global _voyage
    with _lock:
        if _voyage is None:
            _voyage = VoyageClient(api_key=settings.voyage_api_key, max_retries=1,
                                   timeout=VOYAGE_TIMEOUT_S)
    return _voyage


def _not_ready(message: str) -> QueryError:
    return QueryError(503, "retrieval_not_ready", message)


def _ensure_ready(settings, db) -> None:
    global _ready
    if _ready:
        return
    if not settings.mongodb_uri or not settings.voyage_api_key:
        raise _not_ready("MONGODB_URI and VOYAGE_API_KEY must be set in .env.")
    indexes = list(db[schema.EMBEDDINGS_COLLECTION].list_search_indexes(schema.VECTOR_INDEX_NAME))
    if not indexes:
        raise _not_ready("vector_index not found; run Story 2.2 ingestion.")
    if not (indexes[0].get("queryable") or indexes[0].get("status") == "READY"):
        raise _not_ready("vector_index is not READY yet; retry shortly.")
    sample = db[schema.EMBEDDINGS_COLLECTION].find_one({}, {"model": 1, "dimensions": 1})
    if sample is None:
        raise _not_ready("embeddings collection is empty; run Story 2.2 ingestion.")
    if (sample.get("model") != schema.EMBEDDING_MODEL
            or sample.get("dimensions") != schema.EMBEDDING_DIMENSIONS):
        raise _not_ready("Stored embeddings do not match voyage-3.5/1024; re-run ingestion.")
    _ready = True


def effective_filters(request: QueryRequest) -> dict[str, list[str]]:
    """Server-fixed access_level; caller lists only narrow act/status."""
    f = request.filters
    return {
        "act": list(f.act) if f else [],
        "status": list(f.status) if f else [],
        "access_level": list(SERVER_ACCESS_LEVELS),
    }


def _vector_filter(filters: dict[str, list[str]]) -> dict:
    clauses = [{k: {"$in": v}} for k, v in filters.items() if v]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _check_scope(request: QueryRequest, settings) -> None:
    if request.caller_id is not None and request.caller_id != settings.webui_demo_caller_id:
        raise QueryError(422, "invalid_caller_id", "caller_id is not accepted for this server.")
    mode = request.pattern.value
    if request.required_acts is not None:
        raise QueryError(422, "unsupported_option",
                         f"required_acts is not supported by {mode} mode.")
    if request.chapter is not None:
        raise QueryError(422, "unsupported_option", f"chapter is not supported by {mode} mode.")


def _chunk_result(chunk: dict, section: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk["chunk_id"],
        section_id=chunk["section_id"],
        act=section.get("act") or chunk.get("act") or "",
        text=chunk.get("text") or "",
        heading=section.get("heading") or "",
        score=score,
        chunk_index=chunk.get("chunk_index"),
        act_label=section.get("act_label"),
        status=section.get("status") or chunk.get("status"),
        chapter=section.get("chapter"),
        chapter_title=section.get("chapter_title"),
        section_number=section.get("section_number"),
        source_pdf=section.get("source_pdf"),
        source_sha256=section.get("source_sha256"),
        needs_review=section.get("needs_review"),
    )


def _search(settings, db, question: str, limit: int, candidates: int, filters: dict):
    qv = _voyage_client(settings).embed(
        texts=[question], model=schema.EMBEDDING_MODEL, input_type="query"
    ).embeddings[0]
    if len(qv) != schema.EMBEDDING_DIMENSIONS:
        raise _not_ready("Query embedding dimension mismatch (expected 1024).")
    hits = list(db[schema.EMBEDDINGS_COLLECTION].aggregate([
        {"$vectorSearch": {
            "index": schema.VECTOR_INDEX_NAME, "path": "vector", "queryVector": qv,
            "numCandidates": candidates, "limit": limit, "filter": _vector_filter(filters),
        }},
        {"$project": {"_id": 0, "chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]))
    chunks = {c["chunk_id"]: c for c in db[schema.CHUNKS_COLLECTION].find(
        {"chunk_id": {"$in": [h["chunk_id"] for h in hits]}})}
    section_ids = list({c["section_id"] for c in chunks.values()})
    sections = {s["section_id"]: s for s in db[schema.SECTIONS_COLLECTION].find(
        {"section_id": {"$in": section_ids}})}
    return hits, chunks, sections


def run_semantic(request: QueryRequest) -> QueryResult:
    global _ready
    settings = get_settings()
    _check_scope(request, settings)
    filters = effective_filters(request)
    candidates = num_candidates(request.limit)
    try:
        db = _db(settings) if settings.mongodb_uri else None
        if db is None or not settings.voyage_api_key:
            raise _not_ready("MONGODB_URI and VOYAGE_API_KEY must be set in .env.")
        _ensure_ready(settings, db)
        hits, chunks, sections = _search(
            settings, db, request.question, request.limit, candidates, filters)
    except QueryError:
        _ready = False
        raise
    except Exception as exc:  # noqa: BLE001 - any Mongo/Voyage failure; never leak URIs or keys
        _ready = False
        log.warning("semantic retrieval upstream error: %s", type(exc).__name__)
        raise QueryError(502, "retrieval_upstream_error",
                         "Voyage or MongoDB request failed; retry shortly.") from None

    results: list[RetrievedChunk] = []
    unresolved = 0
    for hit in hits:
        chunk = chunks.get(hit["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if chunk is None or section is None:
            unresolved += 1
            continue
        results.append(_chunk_result(chunk, section, float(hit["score"])))
    if hits and not results:
        _ready = False
        raise _not_ready("Vector hits could not be resolved to chunks/sections; re-run ingestion.")

    status = "ok" if results else "no_results"
    message = (f"Returned {len(results)} passage(s). " if results
               else "No passages match the filters. ") + SIMILARITY_NOTE
    return QueryResult(
        pattern=request.pattern.value,
        status=status,
        message=message,
        trace={
            "mode": "semantic",
            "query": request.question,
            "embedding": {"model": schema.EMBEDDING_MODEL, "input_type": "query",
                          "dimensions": schema.EMBEDDING_DIMENSIONS},
            "index": schema.VECTOR_INDEX_NAME,
            "limit": request.limit,
            "num_candidates": candidates,
            "filters": filters,
            "caller_id": settings.webui_demo_caller_id,
            "result_count": len(results),
            "unresolved_hits": unresolved,
        },
        results=results,
    )
