"""Story 3.2: streamed answers, bounded validation, confidence (fake provider, no network)."""

import json

import pytest
from fastapi.testclient import TestClient

from building_with_rag.app import create_app
from building_with_rag.contracts import QueryResult, RetrievedChunk
from building_with_rag.generation import answer as ans
from building_with_rag.routes import chat
from building_with_rag.settings import Settings

GOOD = "Theft is punishable under BNS [E1]."
UNCITED = "Theft is punishable under BNS."
BAD_LABEL = "Theft is punishable under BNS [E9]."
SUPPORTED = json.dumps({"results": [{"claim": 1, "supported": True}]})
DONE = "[" + "DONE" + "]"


def _retrieval() -> QueryResult:
    chunk = RetrievedChunk(chunk_id="c1", section_id="bns:303", act="BNS_2023", text="Theft text",
                           heading="Theft", score=0.9, section_number=303)
    return QueryResult(pattern="semantic", status="ok", message="m", trace={}, results=[chunk])


@pytest.fixture()
def fake(monkeypatch):
    """Scripted provider: `replies` is a list of attempts (str, or an exception to raise)."""
    settings = Settings(generation_api_base_url="http://x", generation_api_key="k",
                        capstone_api_key="", _env_file=None)
    monkeypatch.setattr(ans, "get_settings", lambda: settings)
    monkeypatch.setattr(chat, "get_settings", lambda: settings)
    monkeypatch.setattr(chat, "retrieve", lambda request: _retrieval())
    state = {"replies": [], "stream_calls": 0}

    def stream_chat(messages):
        reply = state["replies"][state["stream_calls"]]
        state["stream_calls"] += 1
        if isinstance(reply, Exception):
            yield "Partial answer "
            raise reply
        yield from (reply[:10], reply[10:])

    monkeypatch.setattr(ans, "_stream_chat", stream_chat)
    monkeypatch.setattr(ans, "_complete", lambda messages: SUPPORTED)
    return state


def _events(fake):
    fake["stream_calls"] = 0
    return list(ans.stream_answer("q", _retrieval()))


def _chat(fake, stream: bool = True) -> tuple[str, str]:
    fake["stream_calls"] = 0
    body = {"model": "rag-semantic", "stream": stream,
            "messages": [{"role": "user", "content": "q"}]}
    raw = TestClient(create_app()).post("/v1/chat/completions", json=body).text
    if not stream:
        return raw, json.loads(raw)["choices"][0]["message"]["content"]
    content = "".join(
        json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        for line in raw.splitlines() if line.startswith("data: {"))
    return raw, content


def test_passing_attempt_is_high_confidence(fake) -> None:
    fake["replies"] = [GOOD]
    events = _events(fake)
    final = events[-1].result
    streamed = "".join(e.text for e in events if e.kind == "text")
    assert fake["stream_calls"] == 1
    assert streamed == ans.DRAFT_PREFIX + final.text
    assert final.outcome == "answered" and final.confidence == "high"
    assert [c.label for c in final.citations] == ["E1"]
    assert final.attempts[0]["status"] == "passed" and final.issues == []


def test_failed_attempt_then_passing_retry(fake) -> None:
    fake["replies"] = [UNCITED, GOOD]
    events = _events(fake)
    final = events[-1].result
    text = "".join(e.text for e in events if e.kind in ("text", "notice"))
    assert text.count(ans.DRAFT_PREFIX) == 2 and "Retrying (attempt 2 of 2)" in text
    assert [a["status"] for a in final.attempts] == ["failed", "passed"]
    assert [i["check"] for i in final.issues] == ["claim_cited"] and final.issues[0]["attempt"] == 1
    assert final.confidence == "high" and final.outcome == "answered"


def test_both_attempts_fail_is_low_confidence(fake) -> None:
    fake["replies"] = [BAD_LABEL, BAD_LABEL]
    final = _events(fake)[-1].result
    assert fake["stream_calls"] == ans.MAX_ATTEMPTS
    assert final.outcome == "malformed" and final.confidence == "low" and final.text == ""
    assert "[E9]" in final.draft_answer and final.low_confidence_reason
    assert [i["attempt"] for i in final.issues] == [1, 2]
    raw, content = _chat(fake)
    assert "DRAFT — low confidence, not the final answer." in content
    assert raw.strip().endswith("data: " + DONE)


def test_provider_failure_mid_stream(fake) -> None:
    fake["replies"] = [ans.ProviderError("timeout")]
    final = _events(fake)[-1].result
    assert final.outcome == "unavailable" and final.confidence is None
    assert final.draft_answer == "Partial answer"
    raw, content = _chat(fake)
    assert content.rstrip().endswith("the text above is an unchecked draft.")
    assert '"finish_reason": "stop"' in raw and raw.strip().endswith("data: " + DONE)


def test_chat_footer_and_json_match_stream(fake) -> None:
    fake["replies"] = [GOOD]
    _, streamed = _chat(fake, stream=True)
    _, plain = _chat(fake, stream=False)
    assert "Evidence check passed — confidence: high" in streamed
    assert "E1 · BNS §303 · Theft · bns:303" in streamed
    assert plain == streamed
