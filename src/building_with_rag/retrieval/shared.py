"""Shared helpers for the decomposition and HyDE modes (Story 5.2).

Thin wrappers over existing semantic helpers and `answer._complete`; no behavior change there.
"""

import json
import logging

from building_with_rag.contracts import RetrievedChunk
from building_with_rag.generation import answer
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.semantic import (
    QueryError,
    _chunk_result,
    _vector_filter,
)

log = logging.getLogger(__name__)

SETTING_NAMES = (
    ("generation_api_base_url", "GENERATION_API_BASE_URL"),
    ("generation_api_key", "GENERATION_API_KEY"),
    ("mongodb_uri", "MONGODB_URI"),
    ("voyage_api_key", "VOYAGE_API_KEY"),
)


def check_settings(settings) -> None:
    """503 naming the first missing setting, before any provider/Voyage/MongoDB call."""
    for attr, env_name in SETTING_NAMES:
        if not str(getattr(settings, attr, "") or "").strip():
            raise QueryError(503, "retrieval_not_ready", f"{env_name} must be set in .env.")


def complete(messages: list[dict]) -> str | None:
    """One bounded model call. Provider failure -> 502; unusable reply shape -> None."""
    try:
        return answer._complete(messages)
    except answer.ProviderError as exc:
        log.warning("generation provider error: %s", exc.code)
        raise QueryError(502, "retrieval_upstream_error",
                         "Generation provider request failed; retry shortly.") from None
    except answer.ReplyError:
        return None


def parse_json(raw: str):
    """Strict JSON after stripping one code fence; raises ValueError."""
    content = raw.strip()
    fence = answer._FENCE.match(content)
    if fence:
        content = fence.group(1).strip()
    return json.loads(content)


def connect(settings):
    """Database handle after the existing readiness checks (vector index, embeddings)."""
    db = semantic._db(settings)
    semantic._ensure_ready(settings, db)
    return db


def embed_texts(settings, texts: list[str], input_type: str) -> list[list[float]]:
    vectors = semantic._voyage_client(settings).embed(
        texts=texts, model=schema.EMBEDDING_MODEL, input_type=input_type).embeddings
    if len(vectors) != len(texts) or any(len(v) != schema.EMBEDDING_DIMENSIONS for v in vectors):
        raise QueryError(503, "retrieval_not_ready",
                         "Embedding dimension mismatch (expected 1024).")
    return vectors


def vector_search(db, vector: list[float], limit: int, filters: dict) -> list[RetrievedChunk]:
    """One $vectorSearch with a precomputed vector; hits resolved to real chunks."""
    hits = list(db[schema.EMBEDDINGS_COLLECTION].aggregate([
        {"$vectorSearch": {
            "index": schema.VECTOR_INDEX_NAME, "path": "vector", "queryVector": vector,
            "numCandidates": semantic.num_candidates(limit), "limit": limit,
            "filter": _vector_filter(filters),
        }},
        {"$project": {"_id": 0, "chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]))
    chunks = {c["chunk_id"]: c for c in db[schema.CHUNKS_COLLECTION].find(
        {"chunk_id": {"$in": [h["chunk_id"] for h in hits]}})}
    section_ids = list({c["section_id"] for c in chunks.values()})
    sections = {s["section_id"]: s for s in db[schema.SECTIONS_COLLECTION].find(
        {"section_id": {"$in": section_ids}})}
    out: list[RetrievedChunk] = []
    for hit in hits:
        chunk = chunks.get(hit["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if chunk is not None and section is not None:
            out.append(_chunk_result(chunk, section, float(hit["score"])))
    return out


class upstream_guard:
    """Map Voyage/MongoDB failures like semantic: reset readiness, 502 without secrets."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            return False
        semantic._ready = False
        if isinstance(exc, QueryError):
            return False
        if isinstance(exc, Exception):
            log.warning("retrieval upstream error: %s", exc_type.__name__)
            raise QueryError(502, "retrieval_upstream_error",
                             "Voyage or MongoDB request failed; retry shortly.") from None
        return False


def round_robin(lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Interleave by rank (list 1 rank 1, list 2 rank 1, ...), deduplicated by chunk_id."""
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for rank in range(max((len(lst) for lst in lists), default=0)):
        for lst in lists:
            if rank < len(lst) and lst[rank].chunk_id not in seen:
                seen.add(lst[rank].chunk_id)
                out.append(lst[rank])
    return out
