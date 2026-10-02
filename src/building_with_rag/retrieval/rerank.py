"""Hybrid re-ranking (Story 4.2): hybrid candidates -> one provider re-rank call -> final evidence.

Re-ranking scores rank passages only; they do not prove a passage is correct.
"""

import logging
import math
import time

import httpx

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.registry import Pattern
from building_with_rag.retrieval.hybrid import run_hybrid
from building_with_rag.retrieval.semantic import QueryError, _check_scope, _not_ready
from building_with_rag.settings import get_settings

log = logging.getLogger(__name__)

MAX_CANDIDATE_LIMIT = 20
RERANK_NOTE = "Re-ranking scores rank only and do not prove correctness."
NOT_SENT = "not_sent_to_reranker"
BELOW_LIMIT = "below_return_limit"
_HYBRID_TRACE_KEYS = ("embedding", "semantic", "keyword", "fusion", "contribution",
                      "unresolved_hits")


def _upstream_error() -> QueryError:
    return QueryError(502, "retrieval_upstream_error",
                      "Re-ranking request failed; no re-ranked result returned.")


def validate_settings(settings) -> None:
    """503 retrieval_not_ready naming the offending setting (never its value)."""
    if not settings.rerank_api_key:
        raise _not_ready("RERANK_API_KEY is not set; hybrid-reranked requires it.")
    if not settings.rerank_api_base_url:
        raise _not_ready("RERANK_API_BASE_URL is not set; hybrid-reranked requires it.")
    if not settings.rerank_model_name:
        raise _not_ready("RERANK_MODEL_NAME is not set; hybrid-reranked requires it.")
    cand, send, ret = (settings.rerank_candidate_limit, settings.rerank_send_limit,
                       settings.rerank_return_limit)
    if ret < 1:
        raise _not_ready("RERANK_RETURN_LIMIT must be at least 1.")
    if send < ret:
        raise _not_ready("RERANK_SEND_LIMIT must be at least RERANK_RETURN_LIMIT.")
    if cand < send or cand > MAX_CANDIDATE_LIMIT:
        raise _not_ready(
            f"RERANK_CANDIDATE_LIMIT must be between RERANK_SEND_LIMIT and {MAX_CANDIDATE_LIMIT}.")
    if settings.rerank_request_timeout_seconds < 1:
        raise _not_ready("RERANK_REQUEST_TIMEOUT_SECONDS must be at least 1.")


def parse_reply(body, sent: int) -> tuple[dict[int, float], int | None]:
    """Validate a provider reply -> ({sent index: score}, usage tokens). ValueError if invalid."""
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        raise ValueError("data")
    scores: dict[int, float] = {}
    for item in data:
        index = item.get("index") if isinstance(item, dict) else None
        score = item.get("relevance_score") if isinstance(item, dict) else None
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < sent:
            raise ValueError("index")
        if index in scores:
            raise ValueError("duplicate index")
        if isinstance(score, bool) or not isinstance(score, (int, float)) \
                or not math.isfinite(score):
            raise ValueError("score")
        scores[index] = float(score)
    if len(scores) != sent:  # no top_k was sent: every candidate must be scored
        raise ValueError("incomplete")
    usage = body.get("usage")
    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return scores, tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None


def _call_reranker(settings, question: str, documents: list[str]):
    """The single provider call: POST {base}/rerank. Any failure -> 502; no retries."""
    try:
        response = httpx.post(
            settings.rerank_api_base_url.rstrip("/") + "/rerank",
            headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
            json={"model": settings.rerank_model_name, "query": question,
                  "documents": documents},
            timeout=settings.rerank_request_timeout_seconds,
        )
        response.raise_for_status()
        return parse_reply(response.json(), len(documents))
    except Exception as exc:  # noqa: BLE001 - never leak URL, key or provider body
        log.warning("rerank upstream error: %s", type(exc).__name__)
        raise _upstream_error() from None


def select(candidates: list[RetrievedChunk], scores: dict[int, float], send_limit: int,
           final_limit: int) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Pure selection. `candidates` are in fused_rank order; `scores` map index in the sent
    prefix (first `send_limit`) to relevance. Returns (final results, omitted in fused order)."""
    sent = candidates[:send_limit]
    omitted = [c.model_copy(update={"omitted_reason": NOT_SENT}) for c in candidates[send_limit:]]
    order = sorted(range(len(sent)), key=lambda i: (-scores[i], sent[i].fused_rank or 0, i))
    results: list[RetrievedChunk] = []
    for rank, i in enumerate(order, start=1):
        update = {"rerank_score": scores[i], "rerank_rank": rank}
        if rank <= final_limit:
            results.append(sent[i].model_copy(update={**update, "score": scores[i]}))
        else:
            omitted.append(sent[i].model_copy(update={**update, "omitted_reason": BELOW_LIMIT}))
    omitted.sort(key=lambda c: c.fused_rank or 0)
    return results, omitted


def run_hybrid_reranked(request: QueryRequest) -> QueryResult:
    settings = get_settings()
    _check_scope(request, settings)
    validate_settings(settings)
    hybrid = run_hybrid(request.model_copy(update={
        "pattern": Pattern.HYBRID, "limit": settings.rerank_candidate_limit}))
    candidates = list(hybrid.results)[:settings.rerank_candidate_limit]

    sent = candidates[:settings.rerank_send_limit]
    results: list[RetrievedChunk] = []
    omitted: list[RetrievedChunk] = []
    latency_ms = 0
    tokens = None
    if candidates:
        documents = [f"{c.heading}\n{c.text}" for c in sent]
        started = time.perf_counter()
        scores, tokens = _call_reranker(settings, request.question, documents)
        latency_ms = round((time.perf_counter() - started) * 1000)
        results, omitted = select(candidates, scores, settings.rerank_send_limit,
                                  min(request.limit, settings.rerank_return_limit))

    rerank_trace = {
        "model": settings.rerank_model_name,
        "candidate_limit": settings.rerank_candidate_limit,
        "send_limit": settings.rerank_send_limit,
        "return_limit": settings.rerank_return_limit,
        "candidates": len(candidates), "sent": len(sent), "returned": len(results),
        "omitted_before": len(candidates) - len(sent),
        "omitted_after": len(sent) - len(results),
        "latency_ms": latency_ms,
    }
    if tokens is not None:
        rerank_trace["usage_tokens"] = tokens
    trace = {
        "mode": "hybrid-reranked",
        "query": request.question,
        "filters": hybrid.trace.get("filters"),
        "caller_id": hybrid.trace.get("caller_id"),
        "result_count": len(results),
        "hybrid": {k: hybrid.trace.get(k) for k in _HYBRID_TRACE_KEYS},
        "rerank": rerank_trace,
    }
    message = (f"Returned {len(results)} passage(s) by re-rank order. " if results
               else "No passages match the filters. ") + RERANK_NOTE
    return QueryResult(
        pattern=request.pattern.value,
        status="ok" if results else "no_results",
        message=message,
        trace=trace,
        results=results,
        omitted_candidates=omitted,
    )
