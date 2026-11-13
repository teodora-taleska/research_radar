# Remove unnecessary whitespace, broken characters and extraction artifacts
# Extracted PDF text is often messy. We want to normalize it without
# destroying meaningful paragraph boundaries.

"""Text normalization utilities for extracted PDF content."""

import re

from app.ingestion.pdf_loader import ExtractedPage


def clean_text(text: str) -> str:
    """Normalize extracted text while preserving paragraph boundaries.

    Removes unnecessary whitespace, fixes common line-break hyphenation,
    and normalizes line endings.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")

    # Reconnect words split by a hyphen at the end of a line.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

    # Normalize spaces without removing paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    # Avoid excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_pages(pages: list[ExtractedPage]) -> list[ExtractedPage]:
    """Clean extracted pages while preserving their page numbers."""
    return [
        ExtractedPage(
            page_number=page.page_number,
            text=clean_text(page.text),
        )
        for page in pages
    ]
