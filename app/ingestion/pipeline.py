# Coordinate loading, cleaning, chunking and embedding
"""Orchestration of the PDF ingestion pipeline."""

from dataclasses import dataclass
from uuid import uuid4

from app.ingestion.chunker import TextChunk, chunk_pages
from app.ingestion.pdf_loader import extract_pdf
from app.ingestion.text_cleaner import clean_pages


@dataclass(frozen=True)
class ProcessedDocument:
    """Structured result of processing an uploaded PDF."""

    document_id: str
    filename: str
    page_count: int
    chunks: list[TextChunk]


def process_pdf(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> ProcessedDocument:
    """Extract, clean and chunk an uploaded PDF.

    This function coordinates the individual ingestion components.
    It does not persist the document or its chunks to a database.

    Args:
        pdf_bytes: Raw PDF contents.
        filename: Original uploaded filename.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        A ProcessedDocument containing metadata and extracted chunks.
    """
    extracted_pdf = extract_pdf(pdf_bytes)

    cleaned_pages = clean_pages(extracted_pdf.pages)

    chunks = chunk_pages(
        pages=cleaned_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError("No text chunks could be generated from this PDF.")

    return ProcessedDocument(
        document_id=str(uuid4()),
        filename=filename,
        page_count=extracted_pdf.page_count,
        chunks=chunks,
    )
