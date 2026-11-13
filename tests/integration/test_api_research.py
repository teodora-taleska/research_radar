"""Integration tests for the research question API."""

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_document_service, get_research_service
from app.api.main import app
from app.generation.answer_generator import AnswerGenerator
from app.generation.llm_provider import LLMProviderError
from app.generation.prompts import INSUFFICIENT_EVIDENCE_MARKER
from app.retrieval.retriever import Retriever
from app.services.document_service import DocumentService
from app.services.research_service import ResearchService
from app.vectorstore.chroma_store import ChromaDocumentStore

VOCABULARY = ["patients", "accuracy", "attention"]


class KeywordEmbedder:
    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts):
        return [
            [float(text.lower().count(word)) for word in VOCABULARY] + [0.1]
            for text in texts
        ]


class FakeLLM:
    def __init__(self, reply="Answer [1].", error=None):
        self.reply = reply
        self.error = error

    def generate(self, prompt, system=None):
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def store(tmp_path):
    return ChromaDocumentStore(persist_directory=tmp_path / "chroma")


def _install(store, repository, llm):
    embedder = KeywordEmbedder()
    app.dependency_overrides[get_document_service] = lambda: DocumentService(
        embedder, store, repository
    )
    app.dependency_overrides[get_research_service] = lambda: ResearchService(
        Retriever(embedder, store), AnswerGenerator(llm)
    )
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _pdf(*page_texts: str) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        document.new_page().insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def _upload(client, pdf_bytes, filename="trial.pdf"):
    response = client.post(
        "/documents/upload",
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201
    return response.json()["document_id"]


def test_upload_then_ask_returns_cited_answer(store, repository):
    client = _install(store, repository, FakeLLM("Patients were randomised [1]."))
    _upload(client, _pdf("Introduction to self attention.", "The patients were randomised."))

    response = client.post(
        "/research/ask",
        json={"question": "How were patients assigned?", "top_k": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sufficient_evidence"] is True
    assert body["answer"] == "Patients were randomised [1]."
    assert body["citations"][0]["filename"] == "trial.pdf"
    assert body["citations"][0]["page_number"] == 2
    assert "randomised" in body["citations"][0]["excerpt"]
    assert body["retrieved_passages"][0]["page_number"] == 2
    assert len(body["retrieved_passages"]) == 2


def test_ask_with_no_documents_reports_insufficient_evidence(store, repository):
    client = _install(store, repository, FakeLLM("should not be called"))

    response = client.post("/research/ask", json={"question": "Anything?"})

    assert response.status_code == 200
    body = response.json()
    assert body["sufficient_evidence"] is False
    assert body["citations"] == []
    assert body["retrieved_passages"] == []


def test_model_refusal_is_reported(store, repository):
    client = _install(store, repository, FakeLLM(INSUFFICIENT_EVIDENCE_MARKER))
    _upload(client, _pdf("The patients were randomised."))

    body = client.post("/research/ask", json={"question": "Capital of France?"}).json()

    assert body["sufficient_evidence"] is False
    assert body["citations"] == []
    assert body["retrieved_passages"]


def test_ask_can_be_restricted_to_documents(store, repository):
    client = _install(store, repository, FakeLLM("Answer [1]."))
    _upload(client, _pdf("patients in study one"), "one.pdf")
    second = _upload(client, _pdf("patients in study two"), "two.pdf")

    body = client.post(
        "/research/ask",
        json={"question": "patients?", "document_ids": [second]},
    ).json()

    assert {p["filename"] for p in body["retrieved_passages"]} == {"two.pdf"}


def test_llm_failure_returns_503(store, repository):
    client = _install(store, repository, FakeLLM(error=LLMProviderError("Ollama down")))
    _upload(client, _pdf("The patients were randomised."))

    response = client.post("/research/ask", json={"question": "patients?"})

    assert response.status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "ok", "top_k": 0},
        {"question": "ok", "top_k": 21},
        {},
    ],
)
def test_invalid_requests_return_422(store, repository, payload):
    client = _install(store, repository, FakeLLM())

    assert client.post("/research/ask", json=payload).status_code == 422
