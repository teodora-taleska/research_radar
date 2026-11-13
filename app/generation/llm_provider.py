"""Local text generation provider powered by Ollama."""

import os
import re
from typing import Protocol

import ollama

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove chain-of-thought that reasoning models may include.

    Handles a full ``<think>...</think>`` block as well as output that
    contains only the closing tag, which some Qwen3 builds produce.
    """
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


class LLMProviderError(RuntimeError):
    """Raised when a language model cannot generate a response."""


class LLMProvider(Protocol):
    """Interface for text generation providers."""

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response to a prompt."""
        ...


class OllamaLLMProvider:
    """Generate text locally using an Ollama chat model."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float = 300.0,
        temperature: float = 0.1,
        client: ollama.Client | None = None,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_LLM_MODEL", "qwen3:4b-instruct")

        if not self.model.strip():
            raise ValueError("LLM model name cannot be empty.")

        self.temperature = temperature
        self.client = client or ollama.Client(
            host=host or os.getenv("OLLAMA_HOST"),
            timeout=timeout,
        )

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response to a prompt.

        Thinking is disabled where the model supports it, and any
        reasoning text that is still returned is stripped, so callers
        only ever receive the final answer.

        Args:
            prompt: The user prompt.
            system: Optional system instruction.

        Returns:
            The model's answer as stripped text.

        Raises:
            ValueError: If the prompt is empty.
            LLMProviderError: If Ollama fails or returns no text.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        messages = []
        if system and system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                think=False,
                options={"temperature": self.temperature},
            )
        except Exception as exc:
            raise LLMProviderError(
                f"Failed to generate a response using model "
                f"'{self.model}'. Check that Ollama is running "
                f"and the model is available. Details: {exc}"
            ) from exc

        content = strip_reasoning(response.message.content or "")

        if not content:
            raise LLMProviderError(
                f"Model '{self.model}' returned an empty response."
            )

        return content

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()
