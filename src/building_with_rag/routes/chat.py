"""OpenAI-compatible chat adapter.

Maps the selected rag-<pattern> model to the same retrieve + answer_events path as /v1/query,
with server-side demo caller_id and generate_answer. Only answer writing streams; frames are
role -> content chunks -> stop -> [DONE], no custom SSE events. Errors before streaming use the
OpenAI-style envelope.
"""

import hmac
import json
import time
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from building_with_rag.contracts import (
    ChatCompletionRequest,
    GenerationResult,
    QueryRequest,
    QueryResult,
    SemanticFilters,
)
from building_with_rag.pipeline import QueryError, answer_events, has_answer_path, retrieve
from building_with_rag.registry import MODEL_ID_TO_PATTERN
from building_with_rag.settings import get_settings

router = APIRouter()

_FINISH_STOP = "stop"
_DONE = "[" + "DONE" + "]"
_UNAVAILABLE_AFTER_TEXT = "Answer generation unavailable — the text above is an unchecked draft."
_UNAVAILABLE_NO_TEXT = "Answer generation unavailable — no draft was produced."
_UNCHECKED_AFTER_TEXT = "The answer could not be checked — the text above is an unchecked draft."
_UNCHECKED_NO_TEXT = "The model reply was invalid, so no answer is shown."


def _error(status: int, message: str, code: str, kind: str = "invalid_request_error") -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"message": message, "type": kind, "code": code}},
    )


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _check_auth(authorization: str | None) -> None:
    key = get_settings().capstone_api_key
    if not key:
        return
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not hmac.compare_digest(supplied.encode(), key.encode()):
        raise _error(401, "Incorrect API key provided.", "invalid_api_key", "invalid_request_error")


def _query_request(request: ChatCompletionRequest) -> QueryRequest:
    pattern = MODEL_ID_TO_PATTERN.get(request.model)
    if pattern is None:
        raise _error(400, f"Model '{request.model}' not found.", "model_not_found")
    latest_user = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if latest_user is None:
        raise _error(400, "At least one user message is required.", "missing_user_message")
    options = request.rag_options
    filters = None
    if options:
        nested = options.filters
        filters = SemanticFilters(
            act=options.act or (nested.act if nested else []),
            status=options.status or (nested.status if nested else []),
            access_level=options.access_level,
        )
    return QueryRequest(
        question=latest_user,
        pattern=pattern,
        caller_id=get_settings().webui_demo_caller_id,  # server-side demo caller
        filters=filters,
        limit=options.limit if options else 5,
        generate_answer=True,  # server-side decision; adapter never trusts client
        required_acts=options.required_acts if options else None,
        chapter=options.chapter if options else None,
    )


def _source_line(c) -> str:
    act = c.act.split("_")[0]
    number = f"§{c.section_number}" if c.section_number is not None else c.section_id
    return f"{c.label} · {act} {number} · {c.heading} · {c.section_id}"


def _footer(g: GenerationResult | None, had_text: bool) -> str:
    if g is None:
        return ""
    if g.outcome == "answered":
        sources = "\n".join(_source_line(c) for c in g.citations)
        return f"\n\nEvidence check passed — confidence: high\n\nSources:\n{sources}"
    if g.confidence == "low":
        details = "\n".join(f"- {i['check']}: {i['detail']}" for i in g.issues)
        return f"\n\nDRAFT — low confidence, not the final answer. {g.low_confidence_reason}\n{details}"
    if g.outcome == "insufficient_evidence":
        reason = str(g.trace.get("reason", ""))[:200]
        return (("\n\n" if had_text else "")
                + "The retrieved passages do not support an answer to this question."
                + (f" {reason}" if reason else ""))
    if g.outcome == "unavailable":
        return "\n\n" + (_UNAVAILABLE_AFTER_TEXT if had_text else _UNAVAILABLE_NO_TEXT)
    return "\n\n" + (_UNCHECKED_AFTER_TEXT if had_text else _UNCHECKED_NO_TEXT)


def _pieces(question: str, result: QueryResult) -> Iterator[str]:
    """Content pieces for one request: streamed text/notices, then the footer."""
    if not has_answer_path(result):
        yield result.message
        return
    final: GenerationResult | None = None
    had_text = False
    for event in answer_events(question, result):
        if event.kind == "final":
            final = event.result
        else:
            had_text = True
            yield event.text
    footer = _footer(final, had_text)
    if footer:
        yield footer


def _chunk(base: dict, n: int, delta: dict, finish: str | None) -> str:
    choices = [{"index": i, "delta": delta, "finish_reason": finish} for i in range(n)]
    return "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": choices}) + "\n\n"


@router.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    query_request = _query_request(request)
    try:
        result = retrieve(query_request)
    except QueryError as exc:
        raise _error(exc.status_code, exc.message, exc.code) from None
    question = query_request.question

    if not request.stream:
        text = "".join(_pieces(question, result))
        return {
            "id": _completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": index,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": _FINISH_STOP,
                }
                for index in range(request.n)
            ],
        }

    def generate():
        base = {"id": _completion_id(), "created": int(time.time()), "model": request.model}
        yield _chunk(base, request.n, {"role": "assistant"}, None)
        for piece in _pieces(question, result):
            yield _chunk(base, request.n, {"content": piece}, None)
        yield _chunk(base, request.n, {}, _FINISH_STOP)
        yield "data: " + _DONE + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
