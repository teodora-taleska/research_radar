
"""Integration tests for the document upload API."""

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_upload_pdf_successfully(sample_pdf_bytes):
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.pdf",
                sample_pdf_bytes,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    result = response.json()

    assert result["filename"] == "sample.pdf"
    assert result["page_count"] == 1
    assert result["chunk_count"] >= 1
    assert result["status"] == "processed"


def test_upload_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "document.txt",
                b"Some text",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415


def test_upload_rejects_invalid_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "invalid.pdf",
                b"This is not a PDF.",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422
