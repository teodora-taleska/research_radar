"""Unit tests for the Retriever."""

import pytest

from app.retrieval.retriever import Retriever
from app.vectorstore.chroma_store import ChromaDocumentStore

VOCABULARY = ["patients", "accuracy", "attention"]


class KeywordEmbedder:
    """Embeds text as keyword counts so similarity is predictable."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.embed_texts([text])[0]

    def embed_texts(self, texts):
        return [
            [float(text.lower().count(word)) for word in VOCABULARY] + [0.1]
            for text in texts
        ]


@pytest.fixture
def embedder():
    return KeywordEmbedder()


@pytest.fixture
def store(tmp_path, embedder):
    store = ChromaDocumentStore(persist_directory=tmp_path / "chroma")
    texts = [
        "patients were randomised",
        "the accuracy was high",
        "self attention layers",
    ]
    store.add_chunks(
        "doc1",
        "paper.pdf",
        [
            {"chunk_index": i, "page_number": i + 1, "content": text}
            for i, text in enumerate(texts)
        ],
        embedder.embed_texts(texts),
    )
    return store


def test_returns_most_similar_chunk_first(store, embedder):
    results = Retriever(embedder, store).retrieve("How many patients?", top_k=3)

    assert results[0].content == "patients were randomised"
    assert results[0].page_number == 1
    assert results[0].distance <= results[1].distance


def test_respects_top_k(store, embedder):
    assert len(Retriever(embedder, store).retrieve("patients", top_k=2)) == 2


def test_embeds_the_question(store, embedder):
    Retriever(embedder, store).retrieve("What was the accuracy?")

    assert embedder.queries == ["What was the accuracy?"]


def test_filters_by_document(store, embedder):
    store.add_chunks(
        "doc2",
        "other.pdf",
        [{"chunk_index": 0, "page_number": 1, "content": "patients again"}],
        embedder.embed_texts(["patients again"]),
    )

    results = Retriever(embedder, store).retrieve(
        "patients", top_k=5, document_ids=["doc2"]
    )

    assert {r.document_id for r in results} == {"doc2"}


def test_empty_store_returns_no_results(tmp_path, embedder):
    empty = ChromaDocumentStore(persist_directory=tmp_path / "empty")

    assert Retriever(embedder, empty).retrieve("patients") == []


@pytest.mark.parametrize("question", ["", "   "])
def test_rejects_empty_question(store, embedder, question):
    with pytest.raises(ValueError):
        Retriever(embedder, store).retrieve(question)


def test_rejects_invalid_top_k(store, embedder):
    with pytest.raises(ValueError):
        Retriever(embedder, store).retrieve("patients", top_k=0)
