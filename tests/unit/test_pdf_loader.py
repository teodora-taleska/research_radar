
"""Unit tests for PDF extraction."""

import pytest

from app.ingestion.pdf_loader import extract_pdf


def test_extract_pdf_returns_page_text(sample_pdf_bytes):
    result = extract_pdf(sample_pdf_bytes)

    assert result.page_count == 1
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert "Research Radar" in result.pages[0].text


def test_extract_pdf_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        extract_pdf(b"")


def test_extract_pdf_rejects_invalid_pdf():
    with pytest.raises(ValueError, match="read"):
        extract_pdf(b"This is not a PDF file.")
