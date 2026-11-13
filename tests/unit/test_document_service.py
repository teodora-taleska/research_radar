"""Unit tests for DocumentService indexing."""

import pytest

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


def test_indexes_all_chunks_with_metadata(store):
    service = DocumentService(FakeEmbeddingProvider(), store)

    indexed = service.index_document(_document(3))

    assert indexed == 3
    assert store.count() == 3
    results = store.search([5.0, 1.0, 0.5, 0.25], top_k=3)
    assert {r.page_number for r in results} == {1, 2, 3}
    assert all(r.filename == "paper.pdf" for r in results)


def test_embeds_in_batches(store):
    provider = FakeEmbeddingProvider()
    service = DocumentService(provider, store, batch_size=2)

    service.index_document(_document(5))

    assert [len(batch) for batch in provider.batches] == [2, 2, 1]


def test_embedding_failure_leaves_store_untouched(store):
    service = DocumentService(FakeEmbeddingProvider(fail=True), store)

    with pytest.raises(EmbeddingProviderError):
        service.index_document(_document(3))

    assert store.count() == 0


def test_store_failure_is_cleaned_up_and_reraised(store, monkeypatch):
    service = DocumentService(FakeEmbeddingProvider(), store)
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


def test_rejects_invalid_batch_size(store):
    with pytest.raises(ValueError):
        DocumentService(FakeEmbeddingProvider(), store, batch_size=0)
