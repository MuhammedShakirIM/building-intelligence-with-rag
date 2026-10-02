"""One shared retrieval -> context -> generation -> validation path for /v1/query and chat."""

from collections.abc import Iterator

from building_with_rag.contracts import GenerationResult, QueryRequest, QueryResult
from building_with_rag.generation.answer import Event, stream_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import QueryError, run_semantic

__all__ = ["QueryError", "answer_events", "attach_generation", "retrieve"]

_OUTCOME_NOTE = {
    "answered": "Answer generated from the cited passages.",
    "insufficient_evidence": "The retrieved passages do not support an answer.",
    "unavailable": "Answer generation was unavailable; retrieved passages are still returned.",
    "malformed": "The model reply was invalid and discarded; retrieved passages are still returned.",
}
_LOW_CONFIDENCE_NOTE = (
    "The answer failed the evidence check and is not validated; "
    "see generation.draft_answer and generation.issues."
)


def retrieve(request: QueryRequest) -> QueryResult:
    """Semantic runs real retrieval; other modes use the shared placeholder dispatch."""
    if request.pattern is Pattern.SEMANTIC:
        return run_semantic(request)
    return QueryResult(**run_pattern(request.pattern, request.question, request.caller_id))


def answer_events(question: str, retrieval: QueryResult) -> Iterator[Event]:
    """The single generation/validation operation; no events for non-semantic modes."""
    if retrieval.pattern == Pattern.SEMANTIC.value:
        yield from stream_answer(question, retrieval)


def attach_generation(question: str, result: QueryResult) -> QueryResult:
    """Drain answer_events and attach the final GenerationResult to the QueryResult."""
    generation: GenerationResult | None = None
    for event in answer_events(question, result):
        if event.kind == "final":
            generation = event.result
    if generation is not None:
        result.generation = generation
        note = (_LOW_CONFIDENCE_NOTE if generation.confidence == "low"
                else _OUTCOME_NOTE[generation.outcome])
        result.message = f"{result.message} {note}"
    return result
