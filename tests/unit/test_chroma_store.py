"""Unit tests for ChromaDocumentStore using an in-memory ChromaDB client."""

import pytest

from app.vectorstore.chroma_store import ChromaDocumentStore, ChromaStoreError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Return a ChromaDocumentStore with an isolated temporary directory."""
    return ChromaDocumentStore(persist_directory=tmp_path / "chroma")


def _make_chunks(n: int, page: int = 1) -> list[dict]:
    return [
        {"chunk_index": i, "page_number": page, "content": f"Chunk {i} text."}
        for i in range(n)
    ]


def _make_embeddings(n: int, dim: int = 4) -> list[list[float]]:
    return [[float(i + j) / 10 for j in range(dim)] for i in range(n)]


# ---------------------------------------------------------------------------
# add_chunks
# ---------------------------------------------------------------------------

class TestAddChunks:
    def test_stores_correct_count(self, store):
        chunks = _make_chunks(3)
        embeddings = _make_embeddings(3)
        result = store.add_chunks("doc1", "paper.pdf", chunks, embeddings)
        assert result == 3
        assert store.count() == 3

    def test_empty_chunks_returns_zero(self, store):
        result = store.add_chunks("doc1", "paper.pdf", [], [])
        assert result == 0
        assert store.count() == 0

    def test_raises_on_length_mismatch(self, store):
        with pytest.raises(ChromaStoreError, match="same length"):
            store.add_chunks("doc1", "paper.pdf", _make_chunks(3), _make_embeddings(2))

    def test_multiple_documents_stored_independently(self, store):
        store.add_chunks("doc1", "a.pdf", _make_chunks(2), _make_embeddings(2))
        store.add_chunks("doc2", "b.pdf", _make_chunks(4), _make_embeddings(4))
        assert store.count() == 6


# ---------------------------------------------------------------------------
# document_exists
# ---------------------------------------------------------------------------

class TestDocumentExists:
    def test_returns_false_for_unknown_document(self, store):
        assert store.document_exists("nonexistent") is False

    def test_returns_true_after_adding_chunks(self, store):
        store.add_chunks("doc1", "paper.pdf", _make_chunks(2), _make_embeddings(2))
        assert store.document_exists("doc1") is True


# ---------------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------------

class TestDeleteDocument:
    def test_deletes_all_chunks_for_document(self, store):
        store.add_chunks("doc1", "paper.pdf", _make_chunks(3), _make_embeddings(3))
        removed = store.delete_document("doc1")
        assert removed == 3
        assert store.count() == 0
        assert store.document_exists("doc1") is False

    def test_delete_does_not_affect_other_documents(self, store):
        store.add_chunks("doc1", "a.pdf", _make_chunks(2), _make_embeddings(2))
        store.add_chunks("doc2", "b.pdf", _make_chunks(3), _make_embeddings(3))
        store.delete_document("doc1")
        assert store.count() == 3
        assert store.document_exists("doc2") is True

    def test_delete_nonexistent_document_returns_zero(self, store):
        result = store.delete_document("ghost")
        assert result == 0


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------

class TestListDocuments:
    def test_empty_store_returns_empty_list(self, store):
        assert store.list_documents() == []

    def test_returns_one_entry_per_document(self, store):
        store.add_chunks("doc1", "a.pdf", _make_chunks(2), _make_embeddings(2))
        store.add_chunks("doc2", "b.pdf", _make_chunks(3), _make_embeddings(3))
        docs = store.list_documents()
        assert len(docs) == 2

    def test_chunk_counts_are_correct(self, store):
        store.add_chunks("doc1", "a.pdf", _make_chunks(2), _make_embeddings(2))
        store.add_chunks("doc2", "b.pdf", _make_chunks(5), _make_embeddings(5))
        docs = {d["document_id"]: d for d in store.list_documents()}
        assert docs["doc1"]["chunk_count"] == 2
        assert docs["doc2"]["chunk_count"] == 5

    def test_filenames_are_preserved(self, store):
        store.add_chunks("doc1", "my_paper.pdf", _make_chunks(1), _make_embeddings(1))
        docs = store.list_documents()
        assert docs[0]["filename"] == "my_paper.pdf"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_returns_top_k_results(self, store):
        store.add_chunks("doc1", "paper.pdf", _make_chunks(5), _make_embeddings(5))
        results = store.search(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=3)
        assert len(results) <= 3

    def test_result_fields_are_populated(self, store):
        store.add_chunks("doc1", "paper.pdf", _make_chunks(2), _make_embeddings(2))
        results = store.search(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=1)
        assert len(results) == 1
        chunk = results[0]
        assert chunk.document_id == "doc1"
        assert chunk.filename == "paper.pdf"
        assert isinstance(chunk.content, str)
        assert isinstance(chunk.distance, float)

    def test_filter_by_document_id(self, store):
        store.add_chunks("doc1", "a.pdf", _make_chunks(3), _make_embeddings(3))
        store.add_chunks("doc2", "b.pdf", _make_chunks(3), _make_embeddings(3))
        results = store.search(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            top_k=10,
            document_ids=["doc1"],
        )
        assert all(r.document_id == "doc1" for r in results)

    def test_filter_by_multiple_document_ids(self, store):
        store.add_chunks("doc1", "a.pdf", _make_chunks(2), _make_embeddings(2))
        store.add_chunks("doc2", "b.pdf", _make_chunks(2), _make_embeddings(2))
        store.add_chunks("doc3", "c.pdf", _make_chunks(2), _make_embeddings(2))
        results = store.search(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            top_k=10,
            document_ids=["doc1", "doc2"],
        )
        assert all(r.document_id in {"doc1", "doc2"} for r in results)

    def test_page_numbers_are_preserved(self, store):
        chunks = [{"chunk_index": 0, "page_number": 7, "content": "Important finding."}]
        store.add_chunks("doc1", "paper.pdf", chunks, _make_embeddings(1))
        results = store.search(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=1)
        assert results[0].page_number == 7
