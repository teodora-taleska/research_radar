"""Shared pytest fixtures for Research Radar."""

import pymupdf
import pytest


@pytest.fixture
def sample_pdf_bytes():
    """Create a small PDF in memory for testing."""
    document = pymupdf.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Research Radar PDF extraction test.",
    )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


@pytest.fixture
def repository(tmp_path):
    """Document metadata repository backed by a temporary SQLite file."""
    from app.database.connection import create_db_engine, create_session_factory
    from app.database.repositories import DocumentRepository

    engine = create_db_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    yield DocumentRepository(create_session_factory(engine))
    engine.dispose()
