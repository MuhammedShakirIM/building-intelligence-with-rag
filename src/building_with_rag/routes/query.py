"""Query endpoint: semantic runs real retrieval; other modes use the shared run_pattern path."""

from fastapi import APIRouter, HTTPException

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.generation.answer import generate_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import QueryError, run_semantic

router = APIRouter()

_OUTCOME_NOTE = {
    "answered": "Answer generated from the cited passages.",
    "insufficient_evidence": "The retrieved passages do not support an answer.",
    "unavailable": "Answer generation was unavailable; retrieved passages are still returned.",
    "malformed": "The model reply was invalid and discarded; retrieved passages are still returned.",
}


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    if request.pattern is Pattern.SEMANTIC:
        try:
            result = run_semantic(request)
        except QueryError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": exc.message},
            ) from None
        if request.generate_answer:
            generation = generate_answer(request.question, result)
            result.generation = generation
            result.message = f"{result.message} {_OUTCOME_NOTE[generation.outcome]}"
        return result
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)
