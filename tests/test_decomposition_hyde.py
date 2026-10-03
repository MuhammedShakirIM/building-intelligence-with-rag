"""Offline decomposition and HyDE tests (Story 5.2): fake model, embedding and vector search."""

import json

import pytest
from pydantic import ValidationError

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk, SubquestionEvidence
from building_with_rag.generation import answer
from building_with_rag.generation.context import assemble_context
from building_with_rag.retrieval import decomposition, hyde, shared
from building_with_rag.retrieval.semantic import QueryError

Q = "How does the BNS punishment for murder differ from the IPC punishment for murder?"
SUBS = {"subquestions": [
    {"text": "What is the BNS punishment for murder?", "act": "BNS_2023"},
    {"text": "What is the IPC punishment for murder?", "act": "IPC_1860"}]}
HYPO = "Whoever commits theft of a bicycle shall be punished with imprisonment."


class _Settings:
    mongodb_uri = "x"
    voyage_api_key = "x"
    generation_api_base_url = "http://x"
    generation_api_key = "x"
    generation_model_name = "m"
    webui_demo_caller_id = "demo-public"


def _chunk(cid, act="BNS_2023", score=0.5):
    return RetrievedChunk(chunk_id=cid, section_id=cid, act=act, text=f"text {cid}",
                          heading="h", score=score)


class _Env:
    """Records every fake call so tests can assert nothing ran."""

    def __init__(self, monkeypatch, replies, hits=None):
        self.replies, self.hits = list(replies), hits or {}
        self.embeds, self.searches, self.completions = [], [], 0
        for mod in (decomposition, hyde):
            monkeypatch.setattr(mod, "get_settings", lambda: _Settings())
        monkeypatch.setattr(answer, "_complete", self._complete)
        monkeypatch.setattr(shared, "connect", lambda s: object())
        monkeypatch.setattr(shared, "embed_texts", self._embed)
        monkeypatch.setattr(shared, "vector_search", self._search)

    def _complete(self, messages):
        self.completions += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)

    def _embed(self, settings, texts, input_type):
        self.embeds.append((texts, input_type))
        return [[0.0] * 1024 for _ in texts]

    def _search(self, db, vector, limit, filters):
        self.searches.append((limit, filters))
        return list(self.hits.get(len(self.searches), []))


def _decompose(monkeypatch, replies, hits=None, **kw):
    env = _Env(monkeypatch, replies, hits)
    req = QueryRequest(question=Q, pattern="decomposition", **kw)
    return decomposition.run_decomposition(req), env


STEP_OK = {"steps": [{"n": 1, "sufficient": True, "reason": "ok"},
                     {"n": 2, "sufficient": True, "reason": "ok"}]}
HITS = {1: [_chunk("bns:103"), _chunk("bns:101")], 2: [_chunk("ipc:302", "IPC_1860"),
                                                       _chunk("bns:103")]}


# (a) validation -----------------------------------------------------------------------------

def test_validate_accepts_two():
    accepted, reason = decomposition.validate_subquestions(json.dumps(SUBS), Q)
    assert reason is None and [s["act"] for s in accepted] == ["BNS_2023", "IPC_1860"]


def _subs(*items):
    return json.dumps({"subquestions": [{"text": t, "act": a} for t, a in items]})


@pytest.mark.parametrize("raw", [
    json.dumps({"subquestions": [], "reason": "not a comparison"}),
    "not json",
    json.dumps({**SUBS, "extra": 1}),
    _subs(*[(f"What is the murder punishment variant {i}?", None) for i in range(4)]),
    _subs(("What is the BNS punishment for murder?", None),
          ("what is the  bns punishment for MURDER?", None)),
    _subs(("What is the BNS punishment for murder?", "XYZ")),
    _subs(("murder " + "x" * 300, None)),
    _subs(("Describe quantum entanglement briefly", None)),
])
def test_invalid_decomposition_is_clarify_without_search(monkeypatch, raw):
    result, env = _decompose(monkeypatch, [raw])
    assert result.status == "clarify" and result.results == [] and result.subquestions == []
    assert env.embeds == [] and env.searches == [] and env.completions == 1
    assert result.trace["decomposition"]["rejected_reason"]


# (b) outcomes and (c) shape ---------------------------------------------------------------

def test_all_evidenced_ok_and_result_shape(monkeypatch):
    result, env = _decompose(monkeypatch, [SUBS, STEP_OK], HITS)
    assert result.status == "ok" and len(env.embeds) == 1 and len(env.embeds[0][0]) == 2
    assert env.embeds[0][1] == "query" and all(limit == 2 for limit, _ in env.searches)
    assert env.searches[0][1]["act"] == ["BNS_2023"] and env.searches[1][1]["act"] == ["IPC_1860"]
    ids = [c.chunk_id for c in result.results]
    assert ids == ["bns:103", "ipc:302", "bns:101"]  # round-robin, deduplicated
    assert len(ids) == len(set(ids))
    for step in result.subquestions:
        assert step.status == "evidenced" and all(c in result.results for c in step.results)
    assert result.trace["sufficiency"]["evidenced"] == 2


def test_one_insufficient_is_partial(monkeypatch):
    verdict = {"steps": [{"n": 1, "sufficient": True, "reason": "ok"},
                         {"n": 2, "sufficient": False, "reason": "no IPC text"}]}
    result, _ = _decompose(monkeypatch, [SUBS, verdict], HITS)
    assert result.status == "partial_answer"
    assert result.subquestions[1].status == "insufficient" and result.subquestions[1].reason
    assert result.subquestions[1].subquestion in result.message
    assert result.subquestions[1].results  # kept for inspection


def test_none_evidenced_and_invalid_sufficiency_clarify(monkeypatch):
    none = {"steps": [{"n": 1, "sufficient": False, "reason": "x"},
                      {"n": 2, "sufficient": False, "reason": "y"}]}
    assert _decompose(monkeypatch, [SUBS, none], HITS)[0].status == "clarify"
    result, _ = _decompose(monkeypatch, [SUBS, {"steps": [{"n": 1, "sufficient": True}]}], HITS)
    assert result.status == "clarify" and result.trace["reason"] == "sufficiency check invalid"
    assert result.results  # passages retained for inspection


def test_no_passages_is_no_results_and_empty_step_skips_model(monkeypatch):
    result, env = _decompose(monkeypatch, [SUBS], {})
    assert result.status == "no_results" and env.completions == 1
    result, env = _decompose(monkeypatch, [SUBS, {"steps": [{"n": 1, "sufficient": True}]}],
                             {1: [_chunk("bns:103")]})
    assert result.status == "partial_answer" and env.completions == 2
    assert result.subquestions[1].status == "insufficient"


def test_act_filter_intersection_empty_skips_search(monkeypatch):
    result, env = _decompose(monkeypatch, [SUBS, {"steps": [{"n": 1, "sufficient": True}]}],
                             {1: [_chunk("bns:103")]}, filters={"act": ["BNS_2023"]})
    assert len(env.searches) == 1 and result.subquestions[1].status == "insufficient"


def test_subquestion_evidence_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        SubquestionEvidence(subquestion="q", status="evidenced", bogus=1)


# (d) answer context -----------------------------------------------------------------------

def test_partial_answer_context_has_only_evidenced_passages(monkeypatch):
    verdict = {"steps": [{"n": 1, "sufficient": True, "reason": "ok"},
                         {"n": 2, "sufficient": False, "reason": "no"}]}
    result, _ = _decompose(monkeypatch, [SUBS, verdict], HITS)
    seen = []
    monkeypatch.setattr(pipeline, "stream_answer", lambda q, r: seen.append(r) or iter(()))
    list(pipeline.answer_events(Q, result))
    ids = {e["chunk_id"] for e in assemble_context(seen[0].results).entries}
    assert ids == {c.chunk_id for c in result.subquestions[0].results}
    assert "ipc:302" in {c.chunk_id for c in result.results}  # still returned for inspection


# (e) HyDE ---------------------------------------------------------------------------------

def _hyde(monkeypatch, replies, direct=(), via=(), **kw):
    env = _Env(monkeypatch, replies, {1: list(via)})
    monkeypatch.setattr(hyde, "run_semantic", lambda r: QueryResult(
        pattern="semantic", status="ok", message="m", trace={}, results=list(direct)))
    req = QueryRequest(question="What happens to someone who secretly takes a bike?",
                       pattern="hyde", **kw)
    return hyde.run_hyde(req), env


def test_hyde_merges_and_keeps_text_in_debug_only(monkeypatch):
    direct = [_chunk("a", score=0.4), _chunk("b", score=0.3)]
    via = [_chunk("b", score=0.6), _chunk("c", score=0.5)]
    result, env = _hyde(monkeypatch, [{"text": HYPO}], direct, via)
    assert result.status == "ok" and env.embeds == [([HYPO], "document")]
    assert [c.chunk_id for c in result.results] == ["a", "b", "c"]
    assert next(c for c in result.results if c.chunk_id == "b").score == 0.6
    assert result.hyde_hypothetical_text_debug == HYPO
    assert [c.chunk_id for c in result.hyde_direct_candidates] == ["a", "b"]
    assert [c.chunk_id for c in result.hyde_query_candidates] == ["b", "c"]
    assert HYPO not in json.dumps(result.trace) and HYPO not in result.message
    assert result.trace["hyde"]["hypothetical_chars"] == len(HYPO)
    assert result.trace["hyde"]["final_from"] == {"both": 1, "direct_only": 1, "hyde_only": 1}
    assert all(HYPO not in e["text"] for e in assemble_context(result.results).entries)


@pytest.mark.parametrize("reply", ["not json", {"text": ""}, {"text": "x" * 1501},
                                   {"text": "ok", "extra": 1}])
def test_hyde_invalid_text_is_unavailable(monkeypatch, reply):
    result, env = _hyde(monkeypatch, [reply], direct=[_chunk("a")], via=[_chunk("c")])
    assert result.status == "hyde_unavailable" and result.results == []
    assert result.hyde_direct_candidates == [] and result.hyde_query_candidates == []
    assert result.hyde_hypothetical_text_debug is None and env.embeds == []


# (f) configuration and provider failure ----------------------------------------------------

@pytest.mark.parametrize("mode,runner", [("decomposition", decomposition.run_decomposition),
                                         ("hyde", hyde.run_hyde)])
def test_missing_generation_setting_is_503_and_nothing_called(monkeypatch, mode, runner):
    env = _Env(monkeypatch, [])

    class NoKey(_Settings):
        generation_api_key = ""

    for mod in (decomposition, hyde):
        monkeypatch.setattr(mod, "get_settings", lambda: NoKey())
    with pytest.raises(QueryError) as exc:
        runner(QueryRequest(question=Q, pattern=mode))
    assert exc.value.status_code == 503 and exc.value.code == "retrieval_not_ready"
    assert "GENERATION_API_KEY" in exc.value.message
    assert env.completions == 0 and env.embeds == [] and env.searches == []


def test_provider_failure_is_502(monkeypatch):
    with pytest.raises(QueryError) as exc:
        _decompose(monkeypatch, [answer.ProviderError("timeout")])
    assert exc.value.status_code == 502 and exc.value.code == "retrieval_upstream_error"
    with pytest.raises(QueryError) as exc:
        _hyde(monkeypatch, [answer.ProviderError("http_500")])
    assert exc.value.status_code == 502


# (g) routing and scope ---------------------------------------------------------------------

def test_retrieve_routes_both_modes_and_answer_path(monkeypatch):
    marker = QueryResult(pattern="x", status="ok", message="m", trace={})
    monkeypatch.setattr(pipeline, "run_decomposition", lambda r: marker)
    monkeypatch.setattr(pipeline, "run_hyde", lambda r: marker)
    assert pipeline.retrieve(QueryRequest(question=Q, pattern="decomposition")) is marker
    assert pipeline.retrieve(QueryRequest(question=Q, pattern="hyde")) is marker
    assert {"decomposition", "hyde"} <= pipeline.REAL_PATTERNS

    def has(pattern, status):
        return pipeline.has_answer_path(
            QueryResult(pattern=pattern, status=status, message="m", trace={}))

    assert has("decomposition", "ok") and has("decomposition", "partial_answer")
    assert not has("decomposition", "clarify") and not has("decomposition", "no_results")
    assert has("hyde", "ok")
    assert not has("hyde", "hyde_unavailable") and not has("hyde", "no_results")


@pytest.mark.parametrize("mode,runner", [("decomposition", decomposition.run_decomposition),
                                         ("hyde", hyde.run_hyde)])
@pytest.mark.parametrize("extra", [{"required_acts": ["BNS_2023"]}, {"chapter": "VI"}])
def test_scope_rejects_unsupported_options(monkeypatch, mode, runner, extra):
    env = _Env(monkeypatch, [])
    with pytest.raises(QueryError) as exc:
        runner(QueryRequest(question=Q, pattern=mode, **extra))
    assert exc.value.status_code == 422 and env.completions == 0
