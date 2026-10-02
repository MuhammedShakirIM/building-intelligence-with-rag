"""Query endpoint: shares retrieve + answer_events with the chat adapter."""

from fastapi import APIRouter, HTTPException

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.pipeline import QueryError, attach_generation, retrieve

router = APIRouter()


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    try:
        result = retrieve(request)
    except QueryError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from None
    if request.generate_answer:
        attach_generation(request.question, result)
    return result
