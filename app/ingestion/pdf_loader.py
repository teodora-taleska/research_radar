# Extract text from PDFs, preserving page numbers
"""PDF text extraction utilities.

This module extracts text from individual PDF pages while preserving
their original page numbers.
"""

from dataclasses import dataclass

import pymupdf


@dataclass(frozen=True)
class ExtractedPage:
    """Text extracted from one PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedPDF:
    """Result of extracting a complete PDF document."""

    page_count: int
    pages: list[ExtractedPage]


def extract_pdf(pdf_bytes: bytes) -> ExtractedPDF:
    """Extract text from a PDF provided as bytes.

    Page numbers are one-based to match the numbering users see in
    conventional PDF readers.

    Args:
        pdf_bytes: Raw contents of the uploaded PDF.

    Returns:
        An ExtractedPDF containing the total page count and extracted pages.

    Raises:
        ValueError: If the PDF is invalid, encrypted, empty, or contains
            no extractable text.
    """
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            pages = [
                ExtractedPage(
                    page_number=page_number,
                    text=page.get_text("text"),
                )
                for page_number, page in enumerate(document, start=1)
            ]

            page_count = len(document)

    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError("Could not read the uploaded PDF.") from exc

    if not any(page.text.strip() for page in pages):
        raise ValueError(
            "No extractable text was found. The PDF may contain scanned images."
        )

    return ExtractedPDF(
        page_count=page_count,
        pages=pages,
    )
