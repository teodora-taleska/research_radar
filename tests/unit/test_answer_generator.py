"""Unit tests for prompts and the AnswerGenerator."""

import pytest

from app.generation.answer_generator import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    AnswerGenerator,
    extract_reference_numbers,
)
from app.generation.llm_provider import LLMProviderError
from app.generation.prompts import (
    INSUFFICIENT_EVIDENCE_MARKER,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from app.vectorstore.chroma_store import StoredChunk


class FakeLLM:
    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt, system=None):
        self.calls.append((prompt, system))
        if self.error:
            raise self.error
        return self.reply


def _chunk(index: int, filename="paper.pdf", page=1, content="Some text."):
    return StoredChunk(
        chunk_id=f"doc__chunk_{index}",
        document_id="doc",
        filename=filename,
        page_number=page,
        chunk_index=index,
        content=content,
        distance=0.25,
    )


@pytest.fixture
def chunks():
    return [
        _chunk(0, "a.pdf", 3, "Patients were randomised into two groups."),
        _chunk(1, "b.pdf", 7, "The model reached 91% accuracy."),
    ]


# --- prompts ---------------------------------------------------------------

def test_prompt_numbers_sources_with_document_and_page(chunks):
    prompt = build_user_prompt("How were patients assigned?", chunks)

    assert "[1] (document: a.pdf, page 3)" in prompt
    assert "[2] (document: b.pdf, page 7)" in prompt
    assert "Patients were randomised into two groups." in prompt
    assert "Question: How were patients assigned?" in prompt


def test_system_prompt_defines_the_insufficient_evidence_marker():
    assert INSUFFICIENT_EVIDENCE_MARKER in SYSTEM_PROMPT


# --- citation parsing ------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Claim [1].", [1]),
        ("Claim [1][2] and [2].", [1, 2]),
        ("Claim [1, 3].", [1, 3]),
        ("No citations here.", []),
    ],
)
def test_extract_reference_numbers(text, expected):
    assert extract_reference_numbers(text) == expected


# --- generation ------------------------------------------------------------

def test_returns_answer_with_verified_citations(chunks):
    llm = FakeLLM("Patients were randomised [1]. Accuracy was 91% [2].")

    result = AnswerGenerator(llm).generate("Question?", chunks)

    assert result.sufficient_evidence is True
    assert result.answer == "Patients were randomised [1]. Accuracy was 91% [2]."
    assert [(c.reference, c.filename, c.page_number) for c in result.citations] == [
        (1, "a.pdf", 3),
        (2, "b.pdf", 7),
    ]
    assert result.citations[0].excerpt == "Patients were randomised into two groups."


def test_sends_system_prompt_and_sources_to_llm(chunks):
    llm = FakeLLM("Answer [1].")

    AnswerGenerator(llm).generate("Question?", chunks)

    prompt, system = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "a.pdf" in prompt and "Question?" in prompt


def test_invented_citation_numbers_are_discarded(chunks):
    llm = FakeLLM("Real claim [1]. Invented claim [9]. Zero [0].")

    result = AnswerGenerator(llm).generate("Question?", chunks)

    assert [c.reference for c in result.citations] == [1]


def test_model_declaring_insufficient_evidence(chunks):
    llm = FakeLLM(INSUFFICIENT_EVIDENCE_MARKER)

    result = AnswerGenerator(llm).generate("Question?", chunks)

    assert result.sufficient_evidence is False
    assert result.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.citations == []


def test_answer_without_valid_citations_is_not_accepted(chunks):
    llm = FakeLLM("A confident answer with no citations at all.")

    result = AnswerGenerator(llm).generate("Question?", chunks)

    assert result.sufficient_evidence is False
    assert result.citations == []


def test_no_chunks_skips_the_llm():
    llm = FakeLLM("should not be used")

    result = AnswerGenerator(llm).generate("Question?", [])

    assert result.sufficient_evidence is False
    assert llm.calls == []


@pytest.mark.parametrize("question", ["", "   "])
def test_rejects_empty_question(chunks, question):
    with pytest.raises(ValueError):
        AnswerGenerator(FakeLLM("x")).generate(question, chunks)


def test_llm_errors_propagate(chunks):
    llm = FakeLLM(error=LLMProviderError("Ollama down"))

    with pytest.raises(LLMProviderError):
        AnswerGenerator(llm).generate("Question?", chunks)
