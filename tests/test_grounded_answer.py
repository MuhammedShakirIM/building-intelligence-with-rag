"""Grounded answer claim derivation and empty-context tests (no network)."""

from building_with_rag.contracts import QueryResult
from building_with_rag.generation import answer as ans


def test_derive_claims_extracts_labels_and_bullets() -> None:
    text = "BNS s.303 punishes theft [E1].\n- It is cognizable [E1, E2]\n- Uncited point."
    claims = ans.derive_claims(text)
    assert [c["evidence"] for c in claims] == [["E1"], ["E1", "E2"], []]
    assert claims[0]["text"] == "BNS s.303 punishes theft ."


def test_empty_results_skip_model_call(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("model must not be called")

    monkeypatch.setattr(ans.httpx, "post", boom)
    monkeypatch.setattr(ans.httpx, "stream", boom)
    retrieval = QueryResult(pattern="semantic", status="no_results", message="", trace={})
    result = ans.generate_answer("q", retrieval)
    assert result.outcome == "insufficient_evidence" and result.context_outcome == "empty"
