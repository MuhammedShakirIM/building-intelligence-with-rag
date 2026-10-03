"""Shared API contracts. Later stories extend additively; never rename or add provider variants."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.registry import Pattern


class SemanticFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    act: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)
    access_level: list[str] = Field(default_factory=list)

    @field_validator("act")
    @classmethod
    def _check_act(cls, values: list[str]) -> list[str]:
        return [schema.validate_act(v) for v in values]

    @field_validator("status")
    @classmethod
    def _check_status(cls, values: list[str]) -> list[str]:
        return [schema.validate_status(v) for v in values]

    @field_validator("access_level")
    @classmethod
    def _check_access_level(cls, values: list[str]) -> list[str]:
        return [schema.validate_access_level(v) for v in values]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    pattern: Pattern
    caller_id: str | None = None
    filters: SemanticFilters | None = None
    limit: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = False
    required_acts: list[str] | None = None
    chapter: str | None = None

    @field_validator("question", mode="before")
    @classmethod
    def _trim_question(cls, value):
        return value.strip() if isinstance(value, str) else value


class RetrievedChunk(BaseModel):
    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str
    score: float
    # Optional source details; missing source fields stay None, never guessed.
    chunk_index: int | None = None
    act_label: str | None = None
    status: str | None = None
    chapter: str | None = None
    chapter_title: str | None = None
    section_number: int | None = None
    source_pdf: str | None = None
    source_sha256: str | None = None
    needs_review: bool | None = None
    # Additive hybrid route evidence (Story 4.1); None when the route did not return the chunk.
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None
    fused_rank: int | None = None
    # Additive re-ranking evidence (Story 4.2); None outside hybrid-reranked or when not scored.
    rerank_score: float | None = None
    rerank_rank: int | None = None
    omitted_reason: str | None = None


class StructuredSignals(BaseModel):
    """Validated output of the rule-based structured classifier (Story 5.1)."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal["exact_lookup", "filter", "aggregation"] | None = None
    act: Literal["BNS_2023", "IPC_1860"] | None = None
    section_number: int | None = Field(default=None, ge=1, le=999)
    chapter: str | None = None
    status: Literal["ok", "recommendation", "clarification_needed"]
    reason: str

    @model_validator(mode="after")
    def _ok_means_complete_lookup(self):
        if self.status == "ok" and not (
                self.intent == "exact_lookup" and self.act and self.section_number):
            raise ValueError("status 'ok' requires an exact_lookup with act and section_number.")
        return self


class GenerationClaim(BaseModel):
    text: str
    evidence_labels: list[str] = Field(default_factory=list)


class GenerationCitation(BaseModel):
    label: str
    chunk_id: str
    section_id: str
    act: str
    heading: str = ""
    chapter: str | None = None
    section_number: int | None = None
    source_pdf: str | None = None


class GenerationResult(BaseModel):
    text: str = ""
    model: str | None = None
    # Additive optional fields (Story 3.1).
    outcome: Literal["answered", "insufficient_evidence", "unavailable", "malformed"] | None = None
    claims: list[GenerationClaim] = Field(default_factory=list)
    citations: list[GenerationCitation] = Field(default_factory=list)
    supporting_passages: list[RetrievedChunk] = Field(default_factory=list)
    provider: str | None = None
    trace: dict = Field(default_factory=dict)
    context_outcome: Literal["assembled", "empty"] | None = None
    # Additive optional fields (Story 3.2).
    confidence: Literal["high", "low"] | None = None
    issues: list[dict] = Field(default_factory=list)  # {attempt, check, detail}
    attempts: list[dict] = Field(default_factory=list)  # {attempt, status, chars, latency_ms}
    draft_answer: str = ""
    low_confidence_reason: str = ""


class SubquestionEvidence(BaseModel):
    """One accepted decomposition subquestion with its own retrieved passages (Story 5.2)."""

    model_config = ConfigDict(extra="forbid")

    subquestion: str
    status: Literal["evidenced", "insufficient"]
    results: list[RetrievedChunk] = Field(default_factory=list)
    reason: str | None = None


class QueryResult(BaseModel):
    pattern: str
    status: str
    message: str
    trace: dict
    results: list[RetrievedChunk] = Field(default_factory=list)
    generation: GenerationResult | None = None
    # Additive, empty-by-default fields later modes use:
    omitted_candidates: list[RetrievedChunk] = Field(default_factory=list)
    subquestions: list[SubquestionEvidence] = Field(default_factory=list)
    hyde_direct_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_query_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_hypothetical_text_debug: str | None = None


class ChatRagFilters(BaseModel):
    """Nested filter shape sent by the Open WebUI Pipe."""

    act: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)


class ChatRagOptions(BaseModel):
    pattern: Pattern = Pattern.SEMANTIC
    filters: ChatRagFilters | None = None  # additive; flat act/status still accepted
    act: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)
    access_level: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)
    required_acts: list[str] | None = None
    chapter: str | None = None


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|developer|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    n: int = 1
    rag_options: ChatRagOptions | None = None
