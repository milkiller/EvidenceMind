from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .retriever import SearchResult





@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    page_number: int
    excerpt: str
    source_title: str | None = None
    locator: str | None = None
    version: str | None = None

@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    is_grounded: bool

class AnswerGenerator(Protocol):
    def generate(self, question: str, evidence: Sequence[SearchResult]) -> str:
        """Generate an answer to the question based on the provided evidence."""
        ...


def answer_question(
    question: str,
    evidence: Sequence[SearchResult],
    generator: AnswerGenerator | None,
) -> Answer:
    """Generate an answer to the question based on the provided evidence."""
    if not evidence:
        return Answer(
            text="证据不足，无法基于当前文档回答。",
            citations=(),
            is_grounded=False,
        )

    if generator is None:
        raise ValueError("generator is required when evidence is available")
    answer_text = generator.generate(question, evidence)
    citations = tuple(
        Citation(
            chunk_id=result.chunk_id,
            page_number=result.page_number,
            excerpt=result.text.strip(),
            source_title=getattr(result, "source_title", None),
            locator=getattr(result, "locator", None),
            version=getattr(result, "version", None),
        )
        for result in evidence
    )
    is_grounded = bool(evidence)
    return Answer(text=answer_text, citations=citations, is_grounded=is_grounded)
