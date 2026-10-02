"""Bounded, labelled evidence context built from the results of the real retrieval modes."""

from dataclasses import dataclass, field

from building_with_rag.contracts import RetrievedChunk

MAX_PASSAGES = 5
MAX_CONTEXT_CHARS = 12_000


@dataclass
class ContextBundle:
    entries: list[dict] = field(default_factory=list)
    passages: dict[str, RetrievedChunk] = field(default_factory=dict)  # label -> chunk
    omitted: int = 0
    chars: int = 0

    @property
    def outcome(self) -> str:
        return "assembled" if self.entries else "empty"

    def trace(self) -> dict:
        return {
            "labels": {e["label"]: e["chunk_id"] for e in self.entries},
            "selected": len(self.entries),
            "omitted": self.omitted,
            "chars": self.chars,
        }


def assemble_context(results: list[RetrievedChunk]) -> ContextBundle:
    """Take passages in score order up to the budget; never cut a passage mid-text."""
    bundle = ContextBundle()
    for index, chunk in enumerate(results):
        if (len(bundle.entries) >= MAX_PASSAGES
                or bundle.chars + len(chunk.text) > MAX_CONTEXT_CHARS):
            bundle.omitted = len(results) - index
            break
        label = f"E{len(bundle.entries) + 1}"
        bundle.entries.append({
            "label": label,
            "chunk_id": chunk.chunk_id,
            "section_id": chunk.section_id,
            "act": chunk.act,
            "act_label": chunk.act_label,
            "heading": chunk.heading,
            "chapter": chunk.chapter,
            "section_number": chunk.section_number,
            "status": chunk.status,
            "source_pdf": chunk.source_pdf,
            "needs_review": chunk.needs_review,
            "text": chunk.text,
        })
        bundle.passages[label] = chunk
        bundle.chars += len(chunk.text)
    return bundle
