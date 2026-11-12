# Divide extracted text into manageable chunks

"""Text chunking utilities for extracted scientific papers."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.pdf_loader import ExtractedPage


@dataclass(frozen=True)
class TextChunk:
    """A chunk of document text with its source location."""

    chunk_index: int
    page_number: int
    content: str
    start_index: int | None = None


def chunk_pages(
    pages: list[ExtractedPage],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """Split cleaned pages into overlapping text chunks.

    Each chunk retains its original page number and its position
    within that page.

    Args:
        pages: Cleaned pages extracted from a PDF.
        chunk_size: Maximum chunk size measured in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of TextChunk objects with globally increasing indexes.

    Raises:
        ValueError: If the chunking configuration is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = []

    for page in pages:
        if not page.text.strip():
            continue

        split_documents = splitter.create_documents(
            texts=[page.text],
            metadatas=[{"page_number": page.page_number}],
        )

        for document in split_documents:
            chunks.append(
                TextChunk(
                    chunk_index=len(chunks),
                    page_number=page.page_number,
                    content=document.page_content,
                    start_index=document.metadata.get("start_index"),
                )
            )

    return chunks
