"""Unit tests for the Ollama LLM provider."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.generation.llm_provider import LLMProviderError, OllamaLLMProvider


def _response(text):
    return SimpleNamespace(message=SimpleNamespace(content=text))


@pytest.fixture
def mock_client():
    client = Mock()
    client.chat.return_value = _response("  An answer.  ")
    return client


@pytest.fixture
def provider(mock_client):
    return OllamaLLMProvider(model="qwen3:4b", client=mock_client)


def test_returns_stripped_answer(provider):
    assert provider.generate("What is RAG?") == "An answer."


def test_sends_prompt_with_thinking_disabled(provider, mock_client):
    provider.generate("What is RAG?")

    kwargs = mock_client.chat.call_args.kwargs
    assert kwargs["model"] == "qwen3:4b"
    assert kwargs["think"] is False
    assert kwargs["messages"] == [{"role": "user", "content": "What is RAG?"}]
    assert kwargs["options"] == {"temperature": 0.1}


def test_includes_system_message_first(provider, mock_client):
    provider.generate("Question?", system="Answer only from context.")

    messages = mock_client.chat.call_args.kwargs["messages"]
    assert messages[0] == {
        "role": "system",
        "content": "Answer only from context.",
    }
    assert messages[1]["role"] == "user"


@pytest.mark.parametrize(
    "raw",
    [
        "<think>First I recall RAG...</think>\n\nThe answer.",
        "Some reasoning with no opening tag.\n</think>\n\nThe answer.",
        "The answer.",
    ],
)
def test_strips_reasoning_from_answer(mock_client, raw):
    mock_client.chat.return_value = _response(raw)
    provider = OllamaLLMProvider(model="qwen3:4b", client=mock_client)

    assert provider.generate("Hello?") == "The answer."


def test_reasoning_only_response_is_treated_as_empty(mock_client):
    mock_client.chat.return_value = _response("<think>Hmm.</think>")
    provider = OllamaLLMProvider(model="qwen3:4b", client=mock_client)

    with pytest.raises(LLMProviderError, match="empty response"):
        provider.generate("Hello?")


@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_rejects_empty_prompt(provider, prompt):
    with pytest.raises(ValueError):
        provider.generate(prompt)


def test_wraps_client_errors(mock_client):
    mock_client.chat.side_effect = ConnectionError("refused")
    provider = OllamaLLMProvider(model="qwen3:4b", client=mock_client)

    with pytest.raises(LLMProviderError, match="Ollama is running"):
        provider.generate("Hello?")


@pytest.mark.parametrize("text", ["", "   ", None])
def test_rejects_empty_model_response(mock_client, text):
    mock_client.chat.return_value = _response(text)
    provider = OllamaLLMProvider(model="qwen3:4b", client=mock_client)

    with pytest.raises(LLMProviderError, match="empty response"):
        provider.generate("Hello?")


def test_rejects_empty_model_name(mock_client):
    with pytest.raises(ValueError):
        OllamaLLMProvider(model="  ", client=mock_client)


def test_model_defaults_to_the_instruct_variant(monkeypatch, mock_client):
    monkeypatch.delenv("OLLAMA_LLM_MODEL", raising=False)

    assert OllamaLLMProvider(client=mock_client).model == "qwen3:4b-instruct"


def test_model_defaults_to_environment_variable(monkeypatch, mock_client):
    monkeypatch.setenv("OLLAMA_LLM_MODEL", "qwen3:8b")

    assert OllamaLLMProvider(client=mock_client).model == "qwen3:8b"
