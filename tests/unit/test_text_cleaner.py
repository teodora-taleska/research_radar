
"""Unit tests for PDF text cleaning."""

from app.ingestion.pdf_loader import ExtractedPage
from app.ingestion.text_cleaner import clean_pages, clean_text


def test_clean_text_removes_excessive_whitespace():
    result = clean_text("  Hello    world.  ")

    assert result == "Hello world."


def test_clean_text_preserves_paragraphs():
    result = clean_text("First paragraph.\n\nSecond paragraph.")

    assert result == "First paragraph.\n\nSecond paragraph."


def test_clean_text_reconnects_hyphenated_words():
    result = clean_text("demon-\nstrated")

    assert result == "demonstrated"


def test_clean_pages_preserves_page_numbers():
    pages = [
        ExtractedPage(page_number=3, text="  First page.  "),
        ExtractedPage(page_number=7, text="  Second page.  "),
    ]

    result = clean_pages(pages)

    assert result[0].page_number == 3
    assert result[1].page_number == 7
    assert result[0].text == "First page."
