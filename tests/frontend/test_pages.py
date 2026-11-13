"""Streamlit page tests using a fake backend client."""

from pathlib import Path

import pytest
from api_client import ApiError, ResearchRadarClient
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
PAGE = str(FRONTEND / "pages" / "1_Research_Assistant.py")

DOCS = [
    {
        "document_id": "d1",
        "filename": "trial.pdf",
        "page_count": 4,
        "chunk_count": 9,
        "status": "indexed",
    },
    {
        "document_id": "d2",
        "filename": "other.pdf",
        "page_count": 2,
        "chunk_count": 3,
        "status": "indexed",
    },
]

ANSWER = {
    "question": "By how much?",
    "answer": "Symptoms dropped by 31% [1].",
    "sufficient_evidence": True,
    "citations": [
        {
            "reference": 1,
            "document_id": "d1",
            "filename": "trial.pdf",
            "page_number": 3,
            "chunk_index": 2,
            "excerpt": "The treatment group showed a 31% reduction.",
            "distance": 0.318,
        }
    ],
    "retrieved_passages": [
        {
            "document_id": "d1",
            "filename": "trial.pdf",
            "page_number": 3,
            "chunk_index": 2,
            "content": "The treatment group showed a 31% reduction.",
            "distance": 0.318,
        }
    ],
}


@pytest.fixture
def backend(monkeypatch):
    """Replace the HTTP client with an in-memory fake and record calls."""
    state = {
        "docs": list(DOCS),
        "healthy": True,
        "answer": ANSWER,
        "ask_error": None,
        "asked": [],
        "deleted": [],
    }

    monkeypatch.setattr(
        ResearchRadarClient, "is_healthy", lambda self: state["healthy"]
    )
    monkeypatch.setattr(
        ResearchRadarClient, "list_documents", lambda self: list(state["docs"])
    )

    def ask(self, question, top_k=5, document_ids=None):
        state["asked"].append((question, top_k, document_ids))
        if state["ask_error"]:
            raise ApiError(state["ask_error"])
        return state["answer"]

    def delete(self, document_id):
        state["deleted"].append(document_id)

    monkeypatch.setattr(ResearchRadarClient, "ask", ask)
    monkeypatch.setattr(ResearchRadarClient, "delete_document", delete)
    return state


def _run():
    return AppTest.from_file(PAGE, default_timeout=15).run()


def test_unreachable_backend_shows_error_and_stops(backend):
    backend["healthy"] = False

    at = _run()

    assert "Cannot reach the Research Radar API" in at.error[0].value
    assert len(at.chat_input) == 0


def test_no_documents_prompts_upload_and_disables_chat(backend):
    backend["docs"] = []

    at = _run()

    assert "Upload one or more PDFs" in at.info[0].value
    assert at.chat_input[0].disabled is True


def test_lists_uploaded_papers(backend):
    at = _run()

    assert [c.label for c in at.sidebar.checkbox] == ["trial.pdf", "other.pdf"]
    assert at.chat_input[0].disabled is False


def test_question_shows_answer_with_citation(backend):
    at = _run()

    at.chat_input[0].set_value("By how much?").run()

    assert not at.exception
    assert backend["asked"] == [("By how much?", 5, None)]
    assert any("Symptoms dropped by 31% [1]." in m.value for m in at.markdown)
    labels = [e.label for e in at.expander]
    assert "[1] trial.pdf - page 3" in labels
    assert "All retrieved passages (1)" in labels


def test_insufficient_evidence_is_shown_as_warning(backend):
    backend["answer"] = {
        "question": "x",
        "answer": "The uploaded documents do not contain enough evidence "
        "to answer this question.",
        "sufficient_evidence": False,
        "citations": [],
        "retrieved_passages": [],
    }
    at = _run()

    at.chat_input[0].set_value("Capital of France?").run()

    assert "do not contain enough evidence" in at.warning[0].value
    assert not [e for e in at.expander if e.label.startswith("[")]


def test_unticked_paper_is_excluded_from_search(backend):
    at = _run()
    at.sidebar.checkbox(key="select_d2").uncheck().run()

    at.chat_input[0].set_value("By how much?").run()

    assert backend["asked"][0][2] == ["d1"]


def test_all_papers_unticked_disables_chat(backend):
    at = _run()
    at.sidebar.checkbox(key="select_d1").uncheck()
    at.sidebar.checkbox(key="select_d2").uncheck().run()

    assert at.chat_input[0].disabled is True
    assert "Tick at least one paper" in at.warning[0].value


def test_api_error_during_question_is_shown(backend):
    backend["ask_error"] = "Ollama is down"
    at = _run()

    at.chat_input[0].set_value("By how much?").run()

    assert any("Ollama is down" in e.value for e in at.error)


def test_delete_button_removes_the_paper(backend):
    at = _run()

    at.sidebar.button(key="delete_d1").click().run()

    assert backend["deleted"] == ["d1"]


def test_clear_conversation_empties_history(backend):
    at = _run()
    at.chat_input[0].set_value("By how much?").run()

    clear = next(b for b in at.sidebar.button if b.label == "Clear conversation")
    clear.click().run()

    assert at.session_state["messages"] == []


@pytest.mark.parametrize(
    "path",
    [
        "Home.py",
        "pages/2_Paper_Explorer.py",
        "pages/3_Paper_Comparison.py",
        "pages/4_Evaluation.py",
    ],
)
def test_other_pages_render_without_errors(path):
    at = AppTest.from_file(str(FRONTEND / path), default_timeout=15).run()

    assert not at.exception
