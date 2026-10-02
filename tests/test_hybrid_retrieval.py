"""Offline hybrid retrieval tests (Story 4.1): no network, no MongoDB."""

import pytest

from building_with_rag.contracts import QueryRequest
from building_with_rag.registry import Pattern
from building_with_rag.retrieval import hybrid
from building_with_rag.retrieval.hybrid import RRF_K, fuse


def _hits(*ids):
    return [{"chunk_id": c, "score": 1.0 - i / 10} for i, c in enumerate(ids)]


def test_fuse_both_routes_rank_first_with_rrf_score():
    out = fuse(_hits("a", "b"), _hits("c", "b"), limit=5)
    assert out[0]["chunk_id"] == "b"
    assert out[0]["semantic_rank"] == 2 and out[0]["keyword_rank"] == 2
    assert out[0]["fused_score"] == pytest.approx(2 / (RRF_K + 2))
    assert [e["fused_rank"] for e in out] == [1, 2, 3]


def test_fuse_single_route_leaves_other_none():
    out = {e["chunk_id"]: e for e in fuse(_hits("a"), _hits("c"), limit=5)}
    assert out["a"]["keyword_score"] is None and out["a"]["keyword_rank"] is None
    assert out["c"]["semantic_score"] is None and out["c"]["semantic_rank"] is None


def test_fuse_tie_order_and_limit_deterministic():
    out = fuse(_hits("z", "y"), _hits("x", "w"), limit=3)
    # rank-1 tie: semantic route first, then chunk_id; same for rank 2
    assert [e["chunk_id"] for e in out] == ["z", "x", "y"]
    assert fuse(_hits("z", "y"), _hits("x", "w"), limit=3) == out


class _Coll:
    def __init__(self, indexes=(), docs=()):
        self._indexes, self._docs = list(indexes), list(docs)

    def list_search_indexes(self, name=None):
        return iter(self._indexes)

    def find_one(self, *a, **k):
        return {"model": "voyage-3.5", "dimensions": 1024}

    def find(self, *a, **k):
        return iter(self._docs)

    def aggregate(self, pipeline):
        return iter(self._docs)


class _Settings:
    mongodb_uri = "x"
    voyage_api_key = "y"
    webui_demo_caller_id = "demo-public"


def _patch(monkeypatch, db):
    monkeypatch.setattr(hybrid, "get_settings", lambda: _Settings())
    monkeypatch.setattr(hybrid, "_db", lambda s: db)
    monkeypatch.setattr(hybrid, "_ready", False)


def _request():
    return QueryRequest(question="theft", pattern=Pattern.HYBRID, limit=2)


def test_run_hybrid_sets_fused_fields(monkeypatch):
    chunk = {"chunk_id": "c1", "section_id": "S1", "text": "t", "chunk_index": 0}
    sections = {"S1": {"section_id": "S1", "act": "BNS_2023", "heading": "h"}}
    db = {"embeddings": _Coll([{"queryable": True}]), "chunks": _Coll([{"queryable": True}])}
    _patch(monkeypatch, db)
    monkeypatch.setattr(hybrid, "_search", lambda *a: (
        [{"chunk_id": "c1", "score": 0.9}], {"c1": chunk}, dict(sections)))
    monkeypatch.setattr(hybrid, "_keyword_search",
                        lambda *a: [{"chunk_id": "c1", "section_id": "S1", "score": 4.0}])
    result = hybrid.run_hybrid(_request())
    r = result.results[0]
    assert result.status == "ok" and r.score == r.fused_score and r.fused_rank == 1
    assert (r.semantic_rank, r.keyword_rank, r.semantic_score, r.keyword_score) == (1, 1, 0.9, 4.0)
    assert result.trace["contribution"] == {"both": 1, "semantic_only": 0, "keyword_only": 0}


def test_run_hybrid_missing_keyword_index_not_ready(monkeypatch):
    db = {"embeddings": _Coll([{"queryable": True}]), "chunks": _Coll([])}
    _patch(monkeypatch, db)
    with pytest.raises(hybrid.QueryError) as exc:
        hybrid.run_hybrid(_request())
    assert exc.value.status_code == 503 and exc.value.code == "retrieval_not_ready"
    assert "keyword_index" in exc.value.message
