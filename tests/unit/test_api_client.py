"""Unit tests for the frontend API client."""

import pytest
import requests
from api_client import ApiError, ResearchRadarClient


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse(payload={})
        self.error = error
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.error:
            raise self.error
        return self.response


def _client(response=None, error=None):
    session = FakeSession(response, error)
    return ResearchRadarClient("http://api:8000/", session=session), session


def test_ask_sends_question_and_options():
    client, session = _client(FakeResponse(payload={"answer": "x"}))

    result = client.ask("What?", top_k=3, document_ids=["d1"])

    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "http://api:8000/research/ask")
    assert kwargs["json"] == {
        "question": "What?",
        "top_k": 3,
        "document_ids": ["d1"],
    }
    assert result == {"answer": "x"}


def test_ask_omits_document_ids_when_not_filtering():
    client, session = _client(FakeResponse(payload={}))

    client.ask("What?")

    assert "document_ids" not in session.calls[0][2]["json"]


def test_upload_sends_pdf_file():
    client, session = _client(FakeResponse(201, {"document_id": "d1"}))

    client.upload_document("paper.pdf", b"%PDF")

    kwargs = session.calls[0][2]
    assert kwargs["files"]["file"] == ("paper.pdf", b"%PDF", "application/pdf")


def test_list_and_delete():
    client, session = _client(FakeResponse(payload=[{"document_id": "d1"}]))

    assert client.list_documents() == [{"document_id": "d1"}]
    client.delete_document("d1")

    assert session.calls[1][:2] == ("DELETE", "http://api:8000/documents/d1")


def test_error_detail_from_api_is_reported():
    client, _ = _client(FakeResponse(503, {"detail": "Ollama is down"}))

    with pytest.raises(ApiError, match="Ollama is down"):
        client.ask("What?")


def test_validation_error_list_is_joined():
    detail = [{"msg": "String should have at least 1 character"}]
    client, _ = _client(FakeResponse(422, {"detail": detail}))

    with pytest.raises(ApiError, match="at least 1 character"):
        client.ask("What?")


def test_error_without_json_body_mentions_status():
    client, _ = _client(FakeResponse(500))

    with pytest.raises(ApiError, match="500"):
        client.list_documents()


def test_connection_error_is_explained():
    client, _ = _client(error=requests.exceptions.ConnectionError("refused"))

    with pytest.raises(ApiError, match="Is the backend running"):
        client.list_documents()


def test_timeout_is_explained():
    client, _ = _client(error=requests.exceptions.ReadTimeout("slow"))

    with pytest.raises(ApiError, match="too long"):
        client.ask("What?")


def test_is_healthy_reflects_backend_state():
    healthy, _ = _client(FakeResponse(payload={"status": "healthy"}))
    down, _ = _client(error=requests.exceptions.ConnectionError("refused"))

    assert healthy.is_healthy() is True
    assert down.is_healthy() is False


def test_base_url_defaults_to_environment(monkeypatch):
    monkeypatch.setenv("RESEARCH_RADAR_API_URL", "http://example:9000/")

    assert ResearchRadarClient().base_url == "http://example:9000"
