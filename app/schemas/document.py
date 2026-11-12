
"""Pydantic schemas for document API requests and responses."""

from pydantic import BaseModel, Field


class ChunkPreview(BaseModel):
    """Short preview of a processed document chunk."""

    chunk_index: int
    page_number: int
    character_count: int
    preview: str


class DocumentUploadResponse(BaseModel):
    """Response returned after successfully processing a PDF."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    status: str = "processed"
    chunks: list[ChunkPreview] = Field(default_factory=list)
