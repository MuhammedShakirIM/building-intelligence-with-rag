"""One shared retrieval -> context -> generation -> validation path for /v1/query and chat."""

from collections.abc import Iterator

from building_with_rag.contracts import GenerationResult, QueryRequest, QueryResult
from building_with_rag.generation.answer import Event, stream_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.hybrid import run_hybrid
from building_with_rag.retrieval.rerank import run_hybrid_reranked
from building_with_rag.retrieval.semantic import QueryError, run_semantic

__all__ = ["REAL_PATTERNS", "QueryError", "answer_events", "attach_generation", "retrieve"]

# Modes with real retrieval; they share one context -> generation -> validation path.
REAL_PATTERNS = frozenset(
    {Pattern.SEMANTIC.value, Pattern.HYBRID.value, Pattern.HYBRID_RERANKED.value})

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
    """Semantic, hybrid and hybrid-reranked run real retrieval; others use the placeholder."""
    if request.pattern is Pattern.SEMANTIC:
        return run_semantic(request)
    if request.pattern is Pattern.HYBRID:
        return run_hybrid(request)
    if request.pattern is Pattern.HYBRID_RERANKED:
        return run_hybrid_reranked(request)
    return QueryResult(**run_pattern(request.pattern, request.question, request.caller_id))


def answer_events(question: str, retrieval: QueryResult) -> Iterator[Event]:
    """The single generation/validation operation; no events for placeholder modes."""
    if retrieval.pattern in REAL_PATTERNS:
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
