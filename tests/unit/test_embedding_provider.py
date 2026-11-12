
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.embeddings.provider import (
    EmbeddingProviderError,
    OllamaEmbeddingProvider,
)


@pytest.fixture
def mock_client():
    client = Mock()

    def generate_embeddings(model, input):
        return SimpleNamespace(
            embeddings=[
                [
                    0.1 + i * 0.3,
                    0.2 + i * 0.3,
                    0.3 + i * 0.3,
                ]
                for i, _ in enumerate(input)
            ]
        )

    client.embed.side_effect = generate_embeddings

    return client


@pytest.fixture
def provider(mock_client):
    return OllamaEmbeddingProvider(
        model="embeddinggemma",
        client=mock_client,
    )


def test_embed_single_text(provider, mock_client):
    result = provider.embed_text("Machine learning in healthcare.")

    assert result == [0.1, 0.2, 0.3]
    assert provider.embedding_dimension == 3

    mock_client.embed.assert_called_once_with(
        model="embeddinggemma",
        input=["Machine learning in healthcare."],
    )


def test_embed_multiple_texts(provider):
    texts = [
        "Machine learning in healthcare.",
        "Deep learning for medical imaging.",
    ]

    result = provider.embed_texts(texts)

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]


def test_empty_batch_returns_empty_list(provider, mock_client):
    result = provider.embed_texts([])

    assert result == []
    mock_client.embed.assert_not_called()


def test_rejects_empty_text(provider):
    with pytest.raises(ValueError):
        provider.embed_text("   ")


def test_rejects_non_string_input(provider):
    with pytest.raises(TypeError):
        provider.embed_texts(["valid text", 123])


def test_rejects_string_instead_of_sequence(provider):
    with pytest.raises(TypeError):
        provider.embed_texts("This is not a batch.")


def test_raises_error_when_ollama_fails(provider, mock_client):
    mock_client.embed.side_effect = ConnectionError(
        "Ollama is unavailable."
    )

    with pytest.raises(EmbeddingProviderError):
        provider.embed_text("Test text.")


def test_rejects_inconsistent_embedding_dimensions(
    provider,
    mock_client,
):
    mock_client.embed.side_effect = None
    mock_client.embed.return_value = SimpleNamespace(
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5],
        ]
    )

    with pytest.raises(EmbeddingProviderError):
        provider.embed_texts(["First text", "Second text"])

def test_rejects_dimension_change(provider, mock_client):
    provider.embed_text("First text.")

    mock_client.embed.side_effect = None
    mock_client.embed.return_value = SimpleNamespace(
        embeddings=[[0.1, 0.2, 0.3, 0.4]]
    )

    with pytest.raises(EmbeddingProviderError):
        provider.embed_text("Second text.")

def test_rejects_non_finite_values(provider, mock_client):
    mock_client.embed.side_effect = None
    mock_client.embed.return_value = SimpleNamespace(
        embeddings=[[0.1, float("nan"), 0.3]]
    )

    with pytest.raises(EmbeddingProviderError):
        provider.embed_text("Test text.")