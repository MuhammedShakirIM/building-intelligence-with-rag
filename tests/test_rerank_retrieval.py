"""Offline hybrid re-ranking tests (Story 4.2): no network, no MongoDB."""

import httpx
import pytest

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.generation.context import assemble_context
from building_with_rag.registry import Pattern
from building_with_rag.retrieval import rerank


class _Settings:
    rerank_api_key = "k"
    rerank_api_base_url = "https://example.invalid/v1"
    rerank_model_name = "m"
    rerank_request_timeout_seconds = 5
    rerank_candidate_limit = 4
    rerank_send_limit = 4
    rerank_return_limit = 4
    webui_demo_caller_id = "demo-public"


def _chunks(n):
    return [RetrievedChunk(
        chunk_id=f"c{i}", section_id=f"S{i}", act="BNS_2023", text=f"t{i}", heading=f"h{i}",
        score=1 / i, fused_score=1 / i, fused_rank=i, keyword_rank=i) for i in range(1, n + 1)]


def _request(limit=5):
    return QueryRequest(question="q", pattern=Pattern.HYBRID_RERANKED, limit=limit)


@pytest.fixture()
def setup(monkeypatch):
    calls = {"hybrid": 0, "provider": 0}
    settings = _Settings()
    monkeypatch.setattr(rerank, "get_settings", lambda: settings)

    def fake_hybrid(req):
        calls["hybrid"] += 1
        assert req.pattern is Pattern.HYBRID and req.limit == settings.rerank_candidate_limit
        return QueryResult(pattern="hybrid", status="ok", message="m", results=_chunks(
            settings.rerank_candidate_limit),
            trace={"filters": {}, "caller_id": "demo-public", "embedding": {}, "semantic": {},
                   "keyword": {}, "fusion": {}, "contribution": {}, "unresolved_hits": 0})

    monkeypatch.setattr(rerank, "run_hybrid", fake_hybrid)

    def provider(scores):
        def fake(settings, question, documents):
            calls["provider"] += 1
            assert documents[0] == "h1\nt1" and question == "q"
            return {i: s for i, s in enumerate(scores[:len(documents)])}, 7
        monkeypatch.setattr(rerank, "_call_reranker", fake)

    return settings, calls, provider


def test_reranker_reorders_and_keeps_hybrid_evidence(setup):
    _, calls, provider = setup
    provider([0.1, 0.9, 0.5, 0.7])
    result = rerank.run_hybrid_reranked(_request())
    assert result.status == "ok" and calls["provider"] == 1
    assert [r.chunk_id for r in result.results] == ["c2", "c4", "c3", "c1"]
    assert [r.rerank_rank for r in result.results] == [1, 2, 3, 4]
    assert all(r.score == r.rerank_score for r in result.results)
    assert [r.fused_rank for r in result.results] == [2, 4, 3, 1]
    assert result.omitted_candidates == []
    assert result.trace["rerank"]["usage_tokens"] == 7


def test_omitted_before_and_after_reranking(setup):
    settings, _, provider = setup
    settings.rerank_candidate_limit, settings.rerank_send_limit = 4, 3
    settings.rerank_return_limit = 2
    provider([0.2, 0.9, 0.5])
    result = rerank.run_hybrid_reranked(_request())
    assert [r.chunk_id for r in result.results] == ["c2", "c3"]
    by_id = {c.chunk_id: c for c in result.omitted_candidates}
    assert [c.chunk_id for c in result.omitted_candidates] == ["c1", "c4"]
    assert by_id["c1"].omitted_reason == "below_return_limit"
    assert by_id["c1"].rerank_rank == 3 and by_id["c1"].rerank_score == 0.2
    assert by_id["c4"].omitted_reason == "not_sent_to_reranker"
    assert by_id["c4"].rerank_rank is None and by_id["c4"].rerank_score is None
    t = result.trace["rerank"]
    assert (t["candidates"], t["sent"], t["returned"]) == (4, 3, 2)
    assert (t["omitted_before"], t["omitted_after"]) == (1, 1)


def test_request_limit_caps_final_results(setup):
    _, _, provider = setup
    provider([0.4, 0.3, 0.2, 0.1])
    result = rerank.run_hybrid_reranked(_request(limit=1))
    assert [r.chunk_id for r in result.results] == ["c1"]
    assert len(result.omitted_candidates) == 3


def test_empty_key_503_without_calls(setup):
    settings, calls, provider = setup
    provider([1, 1, 1, 1])
    settings.rerank_api_key = ""
    with pytest.raises(rerank.QueryError) as exc:
        rerank.run_hybrid_reranked(_request())
    assert exc.value.status_code == 503 and exc.value.code == "retrieval_not_ready"
    assert "RERANK_API_KEY" in exc.value.message
    assert calls == {"hybrid": 0, "provider": 0}


@pytest.mark.parametrize("field,value,name", [
    ("rerank_send_limit", 5, "RERANK_SEND_LIMIT"),
    ("rerank_candidate_limit", 21, "RERANK_CANDIDATE_LIMIT"),
    ("rerank_return_limit", 0, "RERANK_RETURN_LIMIT"),
    ("rerank_request_timeout_seconds", 0, "RERANK_REQUEST_TIMEOUT_SECONDS"),
])
def test_invalid_limits_503(setup, field, value, name):
    settings, calls, _ = setup
    setattr(settings, field, value)
    with pytest.raises(rerank.QueryError) as exc:
        rerank.run_hybrid_reranked(_request())
    assert exc.value.status_code == 503 and name in exc.value.message
    assert calls["hybrid"] == 0


class _Resp:
    def __init__(self, status=200, body=None):
        self.status, self.body = status, body

    def raise_for_status(self):
        if self.status >= 400:
            raise httpx.HTTPStatusError("x", request=None, response=None)

    def json(self):
        return self.body


def _ok(*pairs):
    return {"data": [{"index": i, "relevance_score": s} for i, s in pairs]}


def _timeout(*a, **k):
    raise httpx.TimeoutException("t")


@pytest.mark.parametrize("post", [
    _timeout,
    lambda *a, **k: _Resp(500, {}),
    lambda *a, **k: _Resp(200, _ok((0, 0.5), (5, 0.4))),  # index out of range
    lambda *a, **k: _Resp(200, _ok((0, 0.5), (0, 0.4))),  # duplicate index
    lambda *a, **k: _Resp(200, {"data": [{"index": 0}, {"index": 1}]}),  # missing score
    lambda *a, **k: _Resp(200, _ok((0, 0.5))),  # incomplete
    lambda *a, **k: _Resp(200, {"data": []}),
])
def test_provider_failures_502_no_fallback(setup, monkeypatch, post):
    settings, _, _ = setup
    settings.rerank_candidate_limit = settings.rerank_send_limit = 2
    settings.rerank_return_limit = 2
    monkeypatch.setattr(rerank.httpx, "post", post)
    with pytest.raises(rerank.QueryError) as exc:
        rerank.run_hybrid_reranked(_request())
    assert exc.value.status_code == 502 and exc.value.code == "retrieval_upstream_error"
    assert "example.invalid" not in exc.value.message


def test_context_uses_final_results_and_pipeline_routes(setup, monkeypatch):
    settings, _, provider = setup
    settings.rerank_return_limit = 2
    provider([0.1, 0.9, 0.5, 0.7])
    result = pipeline.retrieve(_request())
    bundle = assemble_context(result.results)
    assert [e["chunk_id"] for e in bundle.entries] == ["c2", "c4"]
    assert "hybrid-reranked" in pipeline.REAL_PATTERNS
    assert result.pattern == "hybrid-reranked" and len(result.omitted_candidates) == 2
