"""Offline structured retrieval tests (Story 5.1): no network, fake `sections` collection."""

import pytest

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest
from building_with_rag.generation.context import assemble_context
from building_with_rag.retrieval import structured
from building_with_rag.retrieval.structured import classify

DOC = {"section_id": "bns:103", "act": "BNS_2023", "section_number": 103, "status": "in_force",
       "chapter": "VI", "heading": "Punishment for murder", "text": "Whoever commits murder...",
       "source_pdf": "bns.pdf", "source_status_version": "v1", "needs_review": False,
       "access_level": "public"}


class _Coll:
    def __init__(self, doc=None):
        self.doc, self.calls = doc, []

    def find_one(self, predicate, projection=None):
        self.calls.append((predicate, projection))
        return self.doc


class _Settings:
    mongodb_uri = "x"
    webui_demo_caller_id = "demo-public"


def _run(monkeypatch, question, doc=DOC, **kw):
    coll = _Coll(doc)
    monkeypatch.setattr(structured, "get_settings", lambda: _Settings())
    monkeypatch.setattr(structured, "_db", lambda s: {"sections": coll})
    req = QueryRequest(question=question, pattern="structured", **kw)
    return structured.run_structured(req), coll


@pytest.mark.parametrize("q,act,num", [
    ("What does BNS section 103 say?", "BNS_2023", 103),
    ("IPC sec. 302", "IPC_1860", 302),
    ("Indian Penal Code section 420", "IPC_1860", 420),
])
def test_classify_ok(q, act, num):
    s = classify(q)
    assert (s.status, s.act, s.section_number) == ("ok", act, num)


@pytest.mark.parametrize("q", [
    "What does section 103 say?", "BNS section 103 vs IPC section 302",
    "BNS sections 103 and 105", "BNS section 103A",
])
def test_classify_clarification(q):
    s = classify(q)
    assert s.status == "clarification_needed" and s.intent == "exact_lookup" and s.reason


def test_classify_other_intents():
    assert classify("how many sections are in BNS?").intent == "aggregation"
    assert classify("list all sections in chapter VI").intent == "filter"
    s = classify("What is the punishment for theft?")
    assert (s.status, s.intent) == ("recommendation", None)


def test_invalid_chapter_rejected():
    with pytest.raises(structured.QueryError) as e:
        classify("BNS section 103", chapter="VI; drop")
    assert e.value.status_code == 422


def test_predicate_only_validated_values(monkeypatch):
    result, coll = _run(monkeypatch, 'BNS section 103 {"$ne": null}', chapter="VI")
    assert result.status == "ok"
    predicate, projection = coll.calls[0]
    assert predicate == {"act": "BNS_2023", "section_number": 103,
                         "access_level": {"$in": ["public"]}, "chapter": "VI"}
    assert projection["provenance"] == 0
    chunk = result.results[0]
    assert (chunk.chunk_id, chunk.section_id, chunk.score) == ("bns:103", "bns:103", 1.0)
    assert result.trace["record"]["source_status_version"] == "v1"
    assert result.trace["mongodb_called"] is True


@pytest.mark.parametrize("q", ["What does section 103 say?", "What is theft?", "how many sections?"])
def test_non_ok_makes_no_call(monkeypatch, q):
    result, coll = _run(monkeypatch, q)
    assert coll.calls == [] and result.results == []
    assert result.trace["mongodb_called"] is False and result.status != "ok"


def test_act_filter_conflict_no_call(monkeypatch):
    result, coll = _run(monkeypatch, "BNS section 103", filters={"act": ["IPC_1860"]})
    assert result.status == "clarification_needed" and coll.calls == []


def test_not_found(monkeypatch):
    result, coll = _run(monkeypatch, "BNS section 999", doc=None)
    assert result.status == "not_found" and result.results == []
    assert result.trace["mongodb_called"] is True and len(coll.calls) == 1


def test_routing_and_answer_boundary(monkeypatch):
    ok, _ = _run(monkeypatch, "BNS section 103")
    missing, _ = _run(monkeypatch, "section 103")
    assert not list(pipeline.answer_events("q", missing))
    assert pipeline.has_answer_path(ok) and not pipeline.has_answer_path(missing)
    monkeypatch.setattr(pipeline, "run_structured", lambda r: ok)
    routed = pipeline.retrieve(QueryRequest(question="BNS section 103", pattern="structured"))
    assert routed is ok
    assert assemble_context(ok.results).trace()["labels"] == {"E1": "bns:103"}
