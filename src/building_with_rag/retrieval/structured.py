"""Structured exact retrieval (Story 5.1): rule-based classifier + one read-only `sections` lookup.

Only validated StructuredSignals (act, section_number, optional chapter) reach MongoDB; raw question
text never does. Score 1.0 marks an exact match, not a similarity. No fallback to other modes.
"""

import logging
import re

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk, StructuredSignals
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.retrieval.semantic import QueryError, _db, effective_filters
from building_with_rag.settings import get_settings

log = logging.getLogger(__name__)

PROJECTION = {"_id": 0, "provenance": 0}
_CHAPTER_RE = re.compile(r"^[A-Za-z0-9 .\-]{1,40}$")
_NUM = r"\d+(?:\.\d+)?[A-Za-z]*"
_AGGREGATION_RE = re.compile(r"\b(how many|count|total number)\b", re.IGNORECASE)
_FILTER_RE = re.compile(r"\b(list|all sections|which sections|sections\s+(?:in|under))\b", re.IGNORECASE)
_SECTION_REF_RE = re.compile(
    rf"(?:\bsections?|\bsec\.?|\bs\.|§)\s*({_NUM}(?:\s*(?:,|and|or|&|to)\s*{_NUM})*)", re.IGNORECASE)
_ACT_PATTERNS = {
    "BNS_2023": re.compile(r"\bBNS\b|\bBharatiya\s+Nyaya\s+Sanhita\b", re.IGNORECASE),
    "IPC_1860": re.compile(r"\bIPC\b|\bIndian\s+Penal\s+Code\b", re.IGNORECASE),
}
_ACT_PREFIX = {"BNS_2023": "bns", "IPC_1860": "ipc"}
_OK_NOTE = ("Exact record from the supplied corpus. status and source_status_version describe "
            "the source document and do not claim current legal applicability.")


def _signals(status: str, reason: str, chapter: str | None, intent=None, act=None, number=None):
    return StructuredSignals(intent=intent, act=act, section_number=number, chapter=chapter,
                             status=status, reason=reason)


def classify(question: str, chapter: str | None = None) -> StructuredSignals:
    """Pure rule-based classification; raises QueryError(422) only for an invalid chapter."""
    if chapter is not None and not _CHAPTER_RE.match(chapter):
        raise QueryError(422, "unsupported_option",
                         "chapter must be 1-40 characters: letters, digits, spaces, '.' or '-'.")
    if _AGGREGATION_RE.search(question):
        return _signals("recommendation", "Aggregation (counts) is recognised but not executed "
                        "in structured mode.", chapter, intent="aggregation")
    if _FILTER_RE.search(question):
        return _signals("recommendation", "Filter/listing questions are recognised but not "
                        "executed in structured mode.", chapter, intent="filter")

    runs = [m.group(1) for m in _SECTION_REF_RE.finditer(question)]
    if not runs:
        return _signals("recommendation", "Not an exact section lookup; use semantic or hybrid "
                        "for open questions, or name an act and section (e.g. 'BNS section 103').",
                        chapter)
    tokens = [t for run in runs for t in re.findall(_NUM, run)]
    numbers = {int(t) for t in tokens if t.isdigit() and 1 <= int(t) <= 999}
    acts = [a for a, rx in _ACT_PATTERNS.items() if rx.search(question)]

    def clarify(reason: str):
        return _signals("clarification_needed", reason, chapter, intent="exact_lookup")

    if any(not (t.isdigit() and 1 <= int(t) <= 999) for t in tokens):
        return clarify("Section numbers must be plain integers from 1 to 999 (e.g. 103, not 103A).")
    if len(numbers) > 1:
        return clarify("Several section numbers were given; ask for one section at a time.")
    if not acts:
        return clarify("The act is missing; say BNS or IPC (the act is never guessed).")
    if len(acts) > 1:
        return clarify("Both BNS and IPC were named; ask for one act at a time.")
    return _signals("ok", "Exact section lookup.", chapter, intent="exact_lookup",
                    act=acts[0], number=next(iter(numbers)))


def _check_scope(request: QueryRequest, settings) -> None:
    if request.caller_id is not None and request.caller_id != settings.webui_demo_caller_id:
        raise QueryError(422, "invalid_caller_id", "caller_id is not accepted for this server.")
    if request.required_acts is not None:
        raise QueryError(422, "unsupported_option",
                         "required_acts is not supported by structured mode.")


def _result(request, settings, signals, filters, status, message, called, results=None):
    trace = {
        "mode": "structured",
        "signals": signals.model_dump(),
        "mongodb_called": called,
        "collection": schema.SECTIONS_COLLECTION,
        "filters": filters,
        "caller_id": settings.webui_demo_caller_id,
        "result_count": len(results or []),
    }
    if results:
        doc = results[0]
        trace["record"] = {"section_id": doc.section_id, "status": doc.status,
                           "source_status_version": doc.source_status_version}
    return QueryResult(pattern=request.pattern.value, status=status, message=message,
                       trace=trace, results=[r.chunk for r in results or []])


class _Hit:
    """Chunk plus the source version, for building the trace record."""

    def __init__(self, doc: dict):
        section_id = doc.get("section_id") or (
            f"{_ACT_PREFIX[doc['act']]}:{doc['section_number']}")
        self.chunk = RetrievedChunk(
            chunk_id=section_id, section_id=section_id, act=doc.get("act") or "",
            text=doc.get("text") or "", heading=doc.get("heading") or "", score=1.0,
            act_label=doc.get("act_label"), status=doc.get("status"),
            chapter=doc.get("chapter"), chapter_title=doc.get("chapter_title"),
            section_number=doc.get("section_number"), source_pdf=doc.get("source_pdf"),
            source_sha256=doc.get("source_sha256"), needs_review=doc.get("needs_review"))
        self.section_id = section_id
        self.status = self.chunk.status
        self.source_status_version = doc.get("source_status_version")


def run_structured(request: QueryRequest) -> QueryResult:
    settings = get_settings()
    _check_scope(request, settings)
    signals = classify(request.question, request.chapter)
    filters = effective_filters(request)

    if (signals.status == "ok" and filters["act"] and signals.act not in filters["act"]):
        signals = signals.model_copy(update={
            "status": "clarification_needed",
            "reason": "The act filter excludes the act named in the question."})
    if signals.status != "ok":
        return _result(request, settings, signals, filters, signals.status, signals.reason, False)

    predicate = {"act": signals.act, "section_number": signals.section_number,
                 "access_level": {"$in": filters["access_level"]}}
    if filters["status"]:
        predicate["status"] = {"$in": filters["status"]}
    if signals.chapter is not None:
        predicate["chapter"] = signals.chapter

    if not settings.mongodb_uri:
        raise QueryError(503, "retrieval_not_ready", "MONGODB_URI must be set in .env.")
    try:
        doc = _db(settings)[schema.SECTIONS_COLLECTION].find_one(predicate, PROJECTION)
    except Exception as exc:  # noqa: BLE001 - never leak the URI
        log.warning("structured retrieval upstream error: %s", type(exc).__name__)
        raise QueryError(502, "retrieval_upstream_error",
                         "MongoDB request failed; retry shortly.") from None

    if doc is None:
        message = (f"No record for {signals.act} section {signals.section_number} in this corpus; "
                   "this does not mean the law has no such section.")
        return _result(request, settings, signals, filters, "not_found", message, True)
    return _result(request, settings, signals, filters, "ok", _OK_NOTE, True, [_Hit(doc)])
