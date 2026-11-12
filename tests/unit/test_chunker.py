
"""Unit tests for document chunking."""

import pytest

from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_loader import ExtractedPage


def test_chunk_pages_creates_multiple_chunks():
    pages = [
        ExtractedPage(
            page_number=1,
            text="Research methods and experiments. " * 100,
        )
    ]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_chunk_indexes_are_sequential():
    pages = [
        ExtractedPage(
            page_number=1,
            text="First page. " * 100,
        ),
        ExtractedPage(
            page_number=2,
            text="Second page. " * 100,
        ),
    ]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_empty_pages_are_skipped():
    pages = [
        ExtractedPage(page_number=1, text="   "),
    ]

    assert chunk_pages(pages) == []


def test_invalid_chunk_configuration():
    pages = [ExtractedPage(page_number=1, text="Some text.")]

    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=100, chunk_overlap=100)
