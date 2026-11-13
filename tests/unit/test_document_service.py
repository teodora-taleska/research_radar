"""Unit tests for DocumentService indexing and management."""

import pytest

from app.core.exceptions import DocumentNotFoundError
from app.embeddings.provider import EmbeddingProviderError
from app.ingestion.chunker import TextChunk
from app.ingestion.pipeline import ProcessedDocument
from app.services.document_service import DocumentService
from app.vectorstore.chroma_store import ChromaDocumentStore, ChromaStoreError


class FakeEmbeddingProvider:
    """Deterministic embedder that records every batch it receives."""

    def __init__(self, fail: bool = False) -> None:
        self.batches: list[list[str]] = []
        self.fail = fail

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts):
        if self.fail:
            raise EmbeddingProviderError("Ollama is not running.")
        self.batches.append(list(texts))
        return [[float(len(text)), 1.0, 0.5, 0.25] for text in texts]


def _document(n_chunks: int, document_id: str = "doc1") -> ProcessedDocument:
    return ProcessedDocument(
        document_id=document_id,
        filename="paper.pdf",
        page_count=3,
        chunks=[
            TextChunk(
                chunk_index=i,
                page_number=i + 1,
                content=f"Chunk number {i}",
            )
            for i in range(n_chunks)
        ],
    )


@pytest.fixture
def store(tmp_path):
    return ChromaDocumentStore(persist_directory=tmp_path / "chroma")


@pytest.fixture
def service(store, repository):
    return DocumentService(FakeEmbeddingProvider(), store, repository)


def test_indexes_all_chunks_with_metadata(service, store):
    indexed = service.index_document(_document(3))

    assert indexed == 3
    assert store.count() == 3
    results = store.search([5.0, 1.0, 0.5, 0.25], top_k=3)
    assert {r.page_number for r in results} == {1, 2, 3}
    assert all(r.filename == "paper.pdf" for r in results)


def test_records_document_metadata(service):
    service.index_document(_document(3))

    record = service.get_document("doc1")

    assert record.filename == "paper.pdf"
    assert record.page_count == 3
    assert record.chunk_count == 3
    assert record.status == "indexed"
    assert record.created_at is not None


def test_embeds_in_batches(store, repository):
    provider = FakeEmbeddingProvider()
    service = DocumentService(provider, store, repository, batch_size=2)

    service.index_document(_document(5))

    assert [len(batch) for batch in provider.batches] == [2, 2, 1]


def test_embedding_failure_leaves_everything_untouched(store, repository):
    service = DocumentService(
        FakeEmbeddingProvider(fail=True), store, repository
    )

    with pytest.raises(EmbeddingProviderError):
        service.index_document(_document(3))

    assert store.count() == 0
    assert service.list_documents() == []


def test_vector_store_failure_is_cleaned_up_and_reraised(
    service, store, monkeypatch
):
    store.add_chunks(
        "doc1", "paper.pdf",
        [{"chunk_index": 0, "page_number": 1, "content": "partial"}],
        [[1.0, 1.0, 1.0, 1.0]],
    )

    def failing_add(*args, **kwargs):
        raise ChromaStoreError("disk full")

    monkeypatch.setattr(store, "add_chunks", failing_add)

    with pytest.raises(ChromaStoreError):
        service.index_document(_document(2))

    assert store.document_exists("doc1") is False
    assert service.list_documents() == []


def test_database_failure_removes_the_indexed_vectors(
    service, store, repository, monkeypatch
):
    def failing_add(*args, **kwargs):
        raise RuntimeError("database locked")

    monkeypatch.setattr(repository, "add", failing_add)

    with pytest.raises(RuntimeError):
        service.index_document(_document(2))

    assert store.document_exists("doc1") is False


def test_lists_documents_newest_first(service):
    service.index_document(_document(1, "first"))
    service.index_document(_document(1, "second"))

    assert [d.document_id for d in service.list_documents()] == [
        "second",
        "first",
    ]


def test_get_unknown_document_raises(service):
    with pytest.raises(DocumentNotFoundError):
        service.get_document("ghost")


def test_delete_removes_vectors_and_metadata(service, store):
    service.index_document(_document(3, "doc1"))
    service.index_document(_document(2, "doc2"))

    service.delete_document("doc1")

    assert store.document_exists("doc1") is False
    assert store.document_exists("doc2") is True
    assert [d.document_id for d in service.list_documents()] == ["doc2"]


def test_delete_unknown_document_raises(service):
    with pytest.raises(DocumentNotFoundError):
        service.delete_document("ghost")


def test_failed_vector_delete_keeps_metadata_for_retry(
    service, store, monkeypatch
):
    service.index_document(_document(2))

    def failing_delete(*args, **kwargs):
        raise ChromaStoreError("locked")

    monkeypatch.setattr(store, "delete_document", failing_delete)

    with pytest.raises(ChromaStoreError):
        service.delete_document("doc1")

    assert service.get_document("doc1").document_id == "doc1"


def test_rejects_invalid_batch_size(store, repository):
    with pytest.raises(ValueError):
        DocumentService(FakeEmbeddingProvider(), store, repository, batch_size=0)
