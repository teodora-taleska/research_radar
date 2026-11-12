# Define how text is converted into embeddings.

"""Local text embedding provider powered by Ollama."""

import math
import os
from typing import Protocol, Sequence

import ollama


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider cannot generate embeddings."""


class EmbeddingProvider(Protocol):
    """Interface for text embedding providers."""

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


class OllamaEmbeddingProvider:
    """Generate local text embeddings using an Ollama model."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float = 120.0,
        client: ollama.Client | None = None,
    ) -> None:
        self.model = model or os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            "embeddinggemma",
        )

        if not self.model.strip():
            raise ValueError("Embedding model name cannot be empty.")

        self.client = client or ollama.Client(
            host=host or os.getenv("OLLAMA_HOST"),
            timeout=timeout,
        )

        self._embedding_dimension: int | None = None

    @property
    def embedding_dimension(self) -> int | None:
        """Return the embedding dimension after the first successful request."""
        return self._embedding_dimension

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for one text."""
        return self.embed_texts([text])[0]

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""

        if isinstance(texts, str):
            raise TypeError("Expected a sequence of texts, not a string.")

        if not texts:
            return []

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every input must be a string.")

        if any(not text.strip() for text in texts):
            raise ValueError("Input texts cannot be empty.")

        try:
            response = self.client.embed(
                model=self.model,
                input=list(texts),
            )
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Failed to generate embeddings using model "
                f"'{self.model}'. Check that Ollama is running "
                f"and the model is available. Details: {exc}"
            ) from exc

        embeddings = [
            [float(value) for value in embedding]
            for embedding in response.embeddings
        ]

        if len(embeddings) != len(texts):
            raise EmbeddingProviderError(
                "The number of returned embeddings does not match "
                "the number of input texts."
            )

        if not embeddings or not embeddings[0]:
            raise EmbeddingProviderError(
                "The embedding model returned an empty vector."
            )

        dimensions = len(embeddings[0])

        if any(len(embedding) != dimensions for embedding in embeddings):
            raise EmbeddingProviderError(
                "The model returned embeddings with inconsistent dimensions."
            )

        if any(
            not math.isfinite(value)
            for embedding in embeddings
            for value in embedding
        ):
            raise EmbeddingProviderError(
                "The model returned non-finite embedding values."
            )

        if (
            self._embedding_dimension is not None
            and dimensions != self._embedding_dimension
        ):
            raise EmbeddingProviderError(
                f"Embedding dimension changed from "
                f"{self._embedding_dimension} to {dimensions}."
            )

        self._embedding_dimension = dimensions

        return embeddings

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

