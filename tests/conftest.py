"""Shared pytest fixtures for Research Radar."""

import fitz
import pytest


@pytest.fixture
def sample_pdf_bytes():
    """Create a small PDF in memory for testing."""
    document = fitz.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Research Radar PDF extraction test.",
    )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes
