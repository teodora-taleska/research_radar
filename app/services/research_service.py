"""Orchestration of retrieval and answer generation."""

from dataclasses import dataclass

from app.generation.answer_generator import AnswerGenerator, GeneratedAnswer
from app.retrieval.retriever import Retriever
from app.vectorstore.chroma_store import StoredChunk


@dataclass(frozen=True)
class ResearchResult:
    """An answer together with every passage that was retrieved."""

    question: str
    answer: GeneratedAnswer
    retrieved: list[StoredChunk]


class ResearchService:
    """Answer research questions from the indexed documents."""

    def __init__(
        self,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
    ) -> None:
        self._retriever = retriever
        self._answer_generator = answer_generator

    def ask(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> ResearchResult:
        """Retrieve relevant passages and generate a grounded answer."""
        retrieved = self._retriever.retrieve(
            question,
            top_k=top_k,
            document_ids=document_ids,
        )

        return ResearchResult(
            question=question.strip(),
            answer=self._answer_generator.generate(question, retrieved),
            retrieved=retrieved,
        )
