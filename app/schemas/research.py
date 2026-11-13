"""Pydantic schemas for the research question API."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1000),
]


class AskRequest(BaseModel):
    """A research question about the uploaded documents."""

    question: Question
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] | None = Field(
        default=None,
        description="Restrict the search to these documents. Omit to search all.",
    )


class CitationResponse(BaseModel):
    """A source passage cited by the answer."""

    reference: int
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    excerpt: str
    distance: float


class RetrievedPassage(BaseModel):
    """A passage returned by semantic search, cited or not."""

    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    content: str
    distance: float


class AskResponse(BaseModel):
    """An evidence-grounded answer with its supporting sources."""

    question: str
    answer: str
    sufficient_evidence: bool
    citations: list[CitationResponse]
    retrieved_passages: list[RetrievedPassage]
