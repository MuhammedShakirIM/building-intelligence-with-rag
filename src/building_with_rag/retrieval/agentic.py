"""Bounded agentic RAG (Story 6.1): a fixed, read-only controller over three existing adapters.

Not an LLM planner or loop. Validate -> pick one of three fixed paths -> at most two allowlisted
adapter calls -> inspect every result -> stop with a traced reason. Only real retrieved corpus
passages are evidence; the HyDE hypothetical text and adapter traces (which hold the question)
never enter the controller or its output.
"""

import logging
import re
import time
from types import MappingProxyType

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.registry import Pattern
from building_with_rag.retrieval.hyde import run_hyde
from building_with_rag.retrieval.semantic import (
    QueryError,
    _check_scope,
    effective_filters,
    run_semantic,
)
from building_with_rag.retrieval.structured import classify, run_structured
from building_with_rag.settings import get_settings

log = logging.getLogger(__name__)

# Fixed allowlist; not extensible at runtime.
ALLOWLIST = MappingProxyType({
    "structuredLookup": (run_structured, Pattern.STRUCTURED),
    "semanticRetrieve": (run_semantic, Pattern.SEMANTIC),
    "hydeRetrieve": (run_hyde, Pattern.HYDE),
})
MAX_CALLS = 2
MAX_HYDE_CALLS = 1
LAY_TERMS = ("bike", "scooter", "swindle", "scam", "beat up", "break into", "mobile phone")
ACTS = ("BNS_2023", "IPC_1860")

_BARE_RE = re.compile(r"^\s*(?:sections?|sec\.?|s\.|§)?\s*\d+\s*[?.!]*\s*$", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z]+")
_LAY_RES = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in LAY_TERMS}
_REASON_CODES = (  # classify() reason fragment -> fixed reason_code
    ("plain integers", "section_out_of_range"), ("Several", "several_sections"),
    ("act is missing", "act_missing"), ("Both", "both_acts"))
_GUIDANCE = {
    "bare_section_number": "Name the act (BNS or IPC) and ask a full question, e.g. 'What does BNS section 103 say?'.",
    "act_missing": "Name the act (BNS or IPC); it is never guessed.",
    "both_acts": "Ask about one act at a time.",
    "several_sections": "Ask about one section at a time.",
    "section_out_of_range": "Section numbers must be plain integers from 1 to 999.",
    "listing_or_count": "Counting and listing questions are not supported; ask about one topic or section.",
    "act_filter_conflict": "The act filter excludes the act named in the question.",
    "too_short": "Ask a fuller question (at least three words) or name an act and section.",
}


class _Run:
    def __init__(self, request: QueryRequest, filters: dict, deadline: int, caller_id: str):
        self.request, self.filters, self.deadline, self.caller_id = request, filters, deadline, caller_id
        self.t0 = time.monotonic()
        self.steps: list[dict] = []
        self.hyde_calls = 0
        self.path = "clarify"
        self.validation = {"outcome": "valid", "reason_code": None}
        self.seen: list[RetrievedChunk] = []  # usable passages of the latest call that had any

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def finish(self, status: str, stop_reason: str, message: str,
               results: list[RetrievedChunk] | None = None) -> QueryResult:
        results = results or []
        trace = {
            "mode": "agentic", "caller_id": self.caller_id, "filters": self.filters,
            "limit": self.request.limit, "validation": self.validation, "path": self.path,
            "steps": self.steps, "call_count": len(self.steps), "hyde_calls": self.hyde_calls,
            "deadline_seconds": self.deadline, "elapsed_ms": int(self.elapsed() * 1000),
            "stop_reason": stop_reason, "result_count": len(results),
        }
        return QueryResult(pattern=self.request.pattern.value, status=status, message=message,
                           trace=trace, results=results)

    def ready(self, passages: list[RetrievedChunk]) -> QueryResult:
        return self.finish(
            "ready_for_synthesis", "evidence_adequate",
            f"Agentic mode ran path {self.path} ({len(self.steps)} adapter call(s)); evidence "
            "adequate. Passages are source records; status is not current legal applicability.",
            passages)

    def partial(self, stop_reason: str, results: list[RetrievedChunk] | None = None) -> QueryResult:
        return self.finish(
            "partial_result", stop_reason,
            f"Agentic mode ran path {self.path} ({len(self.steps)} adapter call(s)) and stopped: "
            f"{stop_reason}. No answer is generated.", results)

    def clarify(self, code: str, stop_reason: str = "needs_clarification") -> QueryResult:
        return self.finish("clarify", stop_reason,
                           f"{_GUIDANCE[code]} No search was run." if not self.steps
                           else "The exact lookup needs clarification: name one act and one section.")


def _deadline_seconds(settings) -> int:
    try:
        value = int(str(settings.agentic_deadline_seconds).strip())
    except ValueError:
        value = 0
    if not 1 <= value <= 120:
        raise QueryError(503, "retrieval_not_ready",
                         "AGENTIC_DEADLINE_SECONDS must be an integer from 1 to 120.")
    return value


def _validate(question: str, filters: dict):
    """Pure validation. Returns (reason_code | None, signals)."""
    signals = classify(question)
    words = len(_WORD_RE.findall(question))
    if _BARE_RE.match(question):
        return "bare_section_number", signals
    if signals.intent in ("aggregation", "filter"):
        return "listing_or_count", signals
    if signals.intent == "exact_lookup" and signals.status == "clarification_needed":
        if "act is missing" in signals.reason and words < 3:
            return "bare_section_number", signals
        code = next((c for frag, c in _REASON_CODES if frag in signals.reason), "act_missing")
        return code, signals
    if signals.status == "ok":
        if filters["act"] and signals.act not in filters["act"]:
            return "act_filter_conflict", signals
        return None, signals
    if words < 3:
        return "too_short", signals
    return None, signals


def _usable(chunk: RetrievedChunk) -> bool:
    return bool(chunk.chunk_id.strip() and chunk.section_id.strip() and chunk.text.strip()
                and chunk.act in ACTS)


def _within_filters(results: list[RetrievedChunk], filters: dict, limit: int) -> bool:
    if len(results) > limit:
        return False
    for c in results:
        if filters["act"] and c.act not in filters["act"]:
            return False
        if filters["status"] and c.status not in filters["status"]:
            return False
    return True


def _lay_term(question: str) -> str | None:
    return next((t for t, rx in _LAY_RES.items() if rx.search(question)), None)


def _mentions(chunk: RetrievedChunk, term: str) -> bool:
    return bool(_LAY_RES[term].search(f"{chunk.heading}\n{chunk.text}"))


def _call(run: _Run, name: str):
    """One guarded adapter call. Returns (usable passages, adapter result, stop_reason)."""
    if len(run.steps) >= MAX_CALLS or (name == "hydeRetrieve" and run.hyde_calls >= MAX_HYDE_CALLS):
        return None, None, "budget_exhausted"
    if run.elapsed() >= run.deadline:
        return None, None, "deadline_expired"
    fn, pattern = ALLOWLIST[name]
    step = {"n": len(run.steps) + 1, "adapter": name, "tool_status": None, "returned": 0,
            "usable": 0, "adequate": False, "ms": 0}
    run.steps.append(step)
    if name == "hydeRetrieve":
        run.hyde_calls += 1
    t = time.monotonic()
    try:
        result = fn(run.request.model_copy(update={"pattern": pattern}))
    except QueryError as exc:
        step.update(tool_status=f"http_{exc.status_code}", ms=int((time.monotonic() - t) * 1000))
        return None, None, "tool_unavailable"
    except Exception as exc:  # noqa: BLE001 - adapter failure is a traced stop, never leaks details
        log.warning("agentic adapter error: %s", type(exc).__name__)
        step.update(tool_status="error", ms=int((time.monotonic() - t) * 1000))
        return None, None, "tool_unavailable"
    step["ms"] = int((time.monotonic() - t) * 1000)
    if run.elapsed() > run.deadline:  # late result: discarded, not inspected
        step["tool_status"] = "discarded_late"
        return None, None, "deadline_expired"
    step["tool_status"] = result.status
    if result.status == "hyde_unavailable":
        return None, result, "tool_unavailable"
    step["returned"] = len(result.results)
    if not _within_filters(result.results, run.filters, run.request.limit):
        return None, result, "tool_result_invalid"
    usable = [c for c in result.results if _usable(c)]
    step["usable"] = len(usable)
    if usable:
        run.seen = usable
    return usable, result, None


def run_agentic(request: QueryRequest) -> QueryResult:
    settings = get_settings()
    _check_scope(request, settings)
    deadline = _deadline_seconds(settings)
    filters = effective_filters(request)
    run = _Run(request, filters, deadline, settings.webui_demo_caller_id)

    code, signals = _validate(request.question, filters)
    if code:
        run.validation = {"outcome": "clarify", "reason_code": code}
        return run.clarify(code)

    if signals.status == "ok":  # rule 1: act-qualified section request
        run.path = "exact_lookup"
        usable, result, stop = _call(run, "structuredLookup")
        if stop:
            return run.partial(stop)
        if result.status == "clarification_needed":
            return run.clarify("act_missing")
        if result.status == "not_found":
            return run.partial("record_not_found")
        ok = (result.status == "ok" and len(result.results) == 1 and len(usable) == 1
              and usable[0].act == signals.act and usable[0].section_number == signals.section_number)
        run.steps[-1]["adequate"] = ok
        if ok:
            return run.ready(usable)
        return run.partial("tool_result_invalid" if result.results else "no_usable_evidence")

    term = _lay_term(request.question)
    if term:  # rule 2: vocabulary mismatch
        run.path = "vocab_mismatch"
        usable, result, stop = _call(run, "semanticRetrieve")
        if stop:
            return run.partial(stop)
        direct_ok = any(_mentions(c, term) for c in usable)
        run.steps[-1]["adequate"] = direct_ok
        if direct_ok:
            return run.ready(usable)
        usable, result, stop = _call(run, "hydeRetrieve")
        if stop:
            return run.partial(stop, run.seen if stop != "tool_result_invalid" else None)
        run.steps[-1]["adequate"] = bool(usable)
        if usable:
            return run.ready(usable)
        return run.partial("no_usable_evidence", run.seen)

    run.path = "semantic"  # rule 3: ordinary valid question
    usable, result, stop = _call(run, "semanticRetrieve")
    if stop:
        return run.partial(stop)
    run.steps[-1]["adequate"] = bool(usable)
    if usable:
        return run.ready(usable)
    return run.partial("no_usable_evidence")
