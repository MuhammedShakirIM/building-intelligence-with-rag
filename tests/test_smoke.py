"""Minimal smoke tests: health, models, pattern registry, chat framing (no network)."""

import pytest
from fastapi.testclient import TestClient

from building_with_rag.app import create_app
from building_with_rag.contracts import QueryRequest
from building_with_rag.registry import PATTERN_MODEL_IDS


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_request_accepts_all_six_patterns() -> None:
    for pattern in PATTERN_MODEL_IDS:
        assert QueryRequest(question="What is theft?", pattern=pattern).pattern is pattern


def test_models_lists_six(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert len(ids) == 6 and ids == list(PATTERN_MODEL_IDS.values())


def test_chat_json_clarification(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "rag-structured",
              "messages": [{"role": "user", "content": "What does section 103 say?"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "act is missing" in body["choices"][0]["message"]["content"]


def test_chat_stream_clarification(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "rag-structured",
            "messages": [{"role": "user", "content": "What does section 103 say?"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        raw = b"".join(response.iter_bytes()).decode()
    assert "chat.completion.chunk" in raw
    assert '"finish_reason": "stop"' in raw
    done = "[" + "DONE" + "]"
    assert raw.strip().endswith("data: " + done)


def test_chat_unknown_model_error(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-bogus", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["type"] == "invalid_request_error"
