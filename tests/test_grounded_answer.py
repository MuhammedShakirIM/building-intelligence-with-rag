"""Grounded answer parser/resolver tests (no network)."""

import json

from building_with_rag.contracts import QueryResult, RetrievedChunk
from building_with_rag.generation import answer as ans
from building_with_rag.generation.context import assemble_context

LABELS = {"E1", "E2"}


def _payload(**over) -> str:
    base = {"outcome": "answered", "answer": "BNS s.303 punishes theft.",
            "claims": [{"text": "Theft is punishable.", "evidence": ["E1"]}], "reason": ""}
    return json.dumps({**base, **over})


def _chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"c{i}", section_id=f"bns:{300 + i}", act="BNS_2023",
                          text="text", heading="Theft", score=0.9 - i / 10)


def test_unknown_label_is_malformed() -> None:
    bad = _payload(claims=[{"text": "x", "evidence": ["E9"]}])
    assert ans.parse_model_output(bad, LABELS) is None


def test_non_json_is_malformed() -> None:
    assert ans.parse_model_output("Theft is punishable.", LABELS) is None


def test_answered_without_claims_is_malformed() -> None:
    assert ans.parse_model_output(_payload(claims=[]), LABELS) is None


def test_valid_answer_resolves_citations(monkeypatch) -> None:
    parsed = ans.parse_model_output("```json\n" + _payload() + "\n```", LABELS)
    assert parsed is not None
    bundle = assemble_context([_chunk(0), _chunk(1)])
    result = ans._resolve(parsed, bundle, "m", {})
    assert result.outcome == "answered" and result.text
    assert [c.label for c in result.citations] == ["E1"]
    assert result.citations[0].section_id == "bns:300"
    assert [p.chunk_id for p in result.supporting_passages] == ["c0"]


def test_empty_results_skip_model_call(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("model must not be called")

    monkeypatch.setattr(ans.httpx, "post", boom)
    retrieval = QueryResult(pattern="semantic", status="no_results", message="", trace={})
    result = ans.generate_answer("q", retrieval)
    assert result.outcome == "insufficient_evidence" and result.context_outcome == "empty"
