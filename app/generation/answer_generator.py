"""Generate evidence-grounded answers with verified citations."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.generation.llm_provider import LLMProvider
from app.generation.prompts import (
    INSUFFICIENT_EVIDENCE_MARKER,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from app.vectorstore.chroma_store import StoredChunk

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The uploaded documents do not contain enough evidence "
    "to answer this question."
)

_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Citation:
    """A source passage that supports an answer."""

    reference: int
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    excerpt: str
    distance: float


@dataclass(frozen=True)
class GeneratedAnswer:
    """An answer together with the evidence it relies on."""

    answer: str
    citations: list[Citation]
    sufficient_evidence: bool


def extract_reference_numbers(text: str) -> list[int]:
    """Return cited source numbers in order of first appearance."""
    numbers: list[int] = []
    for group in _CITATION_GROUP.findall(text):
        for part in group.split(","):
            number = int(part.strip())
            if number not in numbers:
                numbers.append(number)
    return numbers


class AnswerGenerator:
    """Answer questions from retrieved passages using an LLM."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def generate(
        self,
        question: str,
        chunks: Sequence[StoredChunk],
    ) -> GeneratedAnswer:
        """Answer a question using only the given passages.

        The answer is reported as insufficient evidence when there are no
        passages, when the model says so, or when the model's answer cites
        no valid source. Citation numbers that do not match a supplied
        passage are discarded, so every returned citation is real.

        Raises:
            ValueError: If the question is empty.
            LLMProviderError: If the language model fails.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not chunks:
            return self._insufficient()

        raw_answer = self._llm.generate(
            build_user_prompt(question, chunks),
            system=SYSTEM_PROMPT,
        )

        if INSUFFICIENT_EVIDENCE_MARKER in raw_answer:
            return self._insufficient()

        citations = [
            Citation(
                reference=number,
                document_id=chunks[number - 1].document_id,
                filename=chunks[number - 1].filename,
                page_number=chunks[number - 1].page_number,
                chunk_index=chunks[number - 1].chunk_index,
                excerpt=chunks[number - 1].content,
                distance=chunks[number - 1].distance,
            )
            for number in extract_reference_numbers(raw_answer)
            if 1 <= number <= len(chunks)
        ]

        if not citations:
            return self._insufficient()

        return GeneratedAnswer(
            answer=raw_answer,
            citations=citations,
            sufficient_evidence=True,
        )

    @staticmethod
    def _insufficient() -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            citations=[],
            sufficient_evidence=False,
        )
