
"""Integration tests for the complete PDF ingestion pipeline."""

from app.ingestion.pipeline import process_pdf


def test_process_pdf_returns_document_with_chunks(sample_pdf_bytes):
    result = process_pdf(
        pdf_bytes=sample_pdf_bytes,
        filename="sample.pdf",
    )

    assert result.document_id
    assert result.filename == "sample.pdf"
    assert result.page_count == 1
    assert len(result.chunks) >= 1

    first_chunk = result.chunks[0]

    assert first_chunk.page_number == 1
    assert "Research Radar" in first_chunk.content
