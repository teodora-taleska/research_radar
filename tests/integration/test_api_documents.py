"""Integration tests for the document upload and management API."""

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_document_service
from app.api.main import app
from app.embeddings.provider import EmbeddingProviderError
from app.services.document_service import DocumentService
from app.vectorstore.chroma_store import ChromaDocumentStore


class FakeEmbeddingProvider:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts):
        if self.fail:
            raise EmbeddingProviderError("Ollama is not running.")
        return [[float(len(text)), 1.0, 0.5, 0.25] for text in texts]


@pytest.fixture
def store(tmp_path):
    return ChromaDocumentStore(persist_directory=tmp_path / "chroma")


@pytest.fixture
def client(store, repository):
    app.dependency_overrides[get_document_service] = lambda: DocumentService(
        FakeEmbeddingProvider(), store, repository
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def _upload(client, pdf_bytes, filename="sample.pdf"):
    return client.post(
        "/documents/upload",
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )


def test_upload_pdf_successfully_and_indexes_chunks(client, store, sample_pdf_bytes):
    response = _upload(client, sample_pdf_bytes)

    assert response.status_code == 201

    result = response.json()

    assert result["filename"] == "sample.pdf"
    assert result["page_count"] == 1
    assert result["chunk_count"] >= 1
    assert result["status"] == "indexed"
    assert store.document_exists(result["document_id"])
    assert store.count() == result["chunk_count"]


def test_upload_returns_503_when_embedding_fails(store, repository, sample_pdf_bytes):
    app.dependency_overrides[get_document_service] = lambda: DocumentService(
        FakeEmbeddingProvider(fail=True), store, repository
    )
    try:
        response = _upload(TestClient(app), sample_pdf_bytes)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert store.count() == 0
    assert repository.list_all() == []


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("document.txt", b"Some text", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_rejects_invalid_pdf(client, store, repository):
    response = _upload(client, b"This is not a PDF.", "invalid.pdf")

    assert response.status_code == 422
    assert store.count() == 0
    assert repository.list_all() == []


def test_list_documents_is_empty_initially(client):
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == []


def test_uploaded_document_appears_in_list_and_detail(client, sample_pdf_bytes):
    document_id = _upload(client, sample_pdf_bytes, "my_paper.pdf").json()["document_id"]

    listing = client.get("/documents").json()
    detail = client.get(f"/documents/{document_id}")

    assert [d["document_id"] for d in listing] == [document_id]
    assert listing[0]["filename"] == "my_paper.pdf"
    assert listing[0]["status"] == "indexed"
    assert listing[0]["created_at"]
    assert detail.status_code == 200
    assert detail.json()["page_count"] == 1


def test_get_unknown_document_returns_404(client):
    assert client.get("/documents/ghost").status_code == 404


def test_delete_removes_document_and_vectors(client, store, sample_pdf_bytes):
    document_id = _upload(client, sample_pdf_bytes).json()["document_id"]

    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 204
    assert client.get("/documents").json() == []
    assert client.get(f"/documents/{document_id}").status_code == 404
    assert store.document_exists(document_id) is False


def test_delete_unknown_document_returns_404(client):
    assert client.delete("/documents/ghost").status_code == 404
