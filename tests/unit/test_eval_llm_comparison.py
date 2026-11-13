"""Tests for the LLM comparison runner."""

import json

import pytest

from app.generation.answer_generator import AnswerGenerator
from app.generation.llm_provider import LLMProviderError
from app.generation.prompts import INSUFFICIENT_EVIDENCE_MARKER
from app.vectorstore.chroma_store import StoredChunk
from evaluation.benchmark import load_benchmark
from evaluation.experiments.llm_comparison import (
    main,
    missing_models,
    run_comparison,
    summarize_model,
)


class ScriptedLLM:
    """Replies with a fixed answer, or raises an error."""

    def __init__(self, reply="", error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply


def _chunk(paper, page, distance=0.3):
    return StoredChunk("id", "doc", paper, page, 0, f"{paper} page {page}", distance)


@pytest.fixture
def benchmark(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(b"%PDF")
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "question": "What did A find?",
                        "relevant": [{"paper": "a.pdf", "pages": [2]}],
                        "expected_answer": "A found X.",
                    },
                    {"id": "q2", "question": "Capital of France?", "answerable": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    return load_benchmark(questions, papers)


def _retrieve(question, top_k):
    if "France" in question:
        return [_chunk("a.pdf", 1, 0.9), _chunk("a.pdf", 2, 0.95)]
    return [_chunk("a.pdf", 2, 0.2), _chunk("a.pdf", 1, 0.4)]


def test_compares_models_on_the_same_retrieved_passages(benchmark):
    good = ScriptedLLM("A found X [1].")
    refuser = ScriptedLLM(INSUFFICIENT_EVIDENCE_MARKER)
    calls = []

    def retrieve(question, top_k):
        calls.append((question, top_k))
        return _retrieve(question, top_k)

    report = run_comparison(
        benchmark,
        retrieve,
        {"good": AnswerGenerator(good), "refuser": AnswerGenerator(refuser)},
        top_k=2,
    )

    assert calls == [("What did A find?", 2), ("Capital of France?", 2)]
    good_summary = report["models"]["good"]["summary"]
    assert good_summary["answer_rate"] == 1.0
    assert good_summary["citation_hit_rate"] == 1.0
    assert good_summary["citation_precision"] == 1.0
    assert good_summary["correct_refusal_rate"] == 0.0
    refuser_summary = report["models"]["refuser"]["summary"]
    assert refuser_summary["answer_rate"] == 0.0
    assert refuser_summary["correct_refusal_rate"] == 1.0
    assert refuser_summary["citation_hit_rate"] is None


def test_records_answers_citations_and_expected_answer(benchmark):
    report = run_comparison(
        benchmark,
        _retrieve,
        {"m": AnswerGenerator(ScriptedLLM("Wrong page [2]. Right page [1]."))},
    )

    first = report["models"]["m"]["answers"][0]
    assert first["id"] == "q1"
    assert first["expected_answer"] == "A found X."
    assert first["answer"] == "Wrong page [2]. Right page [1]."
    assert first["answered"] is True
    assert first["citations"] == [
        {"reference": 2, "paper": "a.pdf", "page": 1, "relevant": False},
        {"reference": 1, "paper": "a.pdf", "page": 2, "relevant": True},
    ]
    assert first["latency_s"] >= 0
    summary = report["models"]["m"]["summary"]
    assert summary["citation_hit_rate"] == 1.0
    assert summary["citation_precision"] == 0.5


def test_answer_citing_the_wrong_page_is_a_citation_miss(benchmark):
    report = run_comparison(
        benchmark, _retrieve, {"m": AnswerGenerator(ScriptedLLM("Claim [2]."))}
    )

    summary = report["models"]["m"]["summary"]
    assert summary["answer_rate"] == 1.0
    assert summary["citation_hit_rate"] == 0.0


def test_llm_errors_are_recorded_and_excluded_from_rates(benchmark):
    broken = AnswerGenerator(ScriptedLLM(error=LLMProviderError("Ollama down")))

    report = run_comparison(benchmark, _retrieve, {"broken": broken})

    summary = report["models"]["broken"]["summary"]
    assert summary["errors"] == 2
    assert summary["answer_rate"] is None
    assert summary["latency_s"] is None
    assert report["models"]["broken"]["answers"][0]["error"] == "Ollama down"


def test_reports_retrieval_distances_for_answerable_and_unanswerable(benchmark):
    report = run_comparison(
        benchmark, _retrieve, {"m": AnswerGenerator(ScriptedLLM("Claim [1]."))}
    )

    distances = report["retrieval_top_distance"]
    assert distances["answerable"]["median"] == pytest.approx(0.2)
    assert distances["unanswerable"]["median"] == pytest.approx(0.9)


def test_prepare_and_progress_hooks(benchmark):
    prepared, progress = [], []

    run_comparison(
        benchmark,
        _retrieve,
        {
            "a": AnswerGenerator(ScriptedLLM("Claim [1].")),
            "b": AnswerGenerator(ScriptedLLM("Claim [1].")),
        },
        prepare=prepared.append,
        progress=lambda model, done, total: progress.append((model, done, total)),
    )

    assert prepared == ["a", "b"]
    assert progress == [("a", 1, 2), ("a", 2, 2), ("b", 1, 2), ("b", 2, 2)]


def test_summarize_model_with_no_answers_has_no_rates():
    summary = summarize_model([])

    assert summary["answer_rate"] is None
    assert summary["correct_refusal_rate"] is None
    assert summary["latency_s"] is None


@pytest.mark.parametrize(
    "requested, installed, expected",
    [
        (["qwen3:4b"], ["qwen3:4b", "embeddinggemma:latest"], []),
        (["embeddinggemma"], ["embeddinggemma:latest"], []),
        (
            ["qwen3:4b-instruct", "qwen3:8b"],
            ["qwen3:4b"],
            ["qwen3:4b-instruct", "qwen3:8b"],
        ),
    ],
)
def test_missing_models(requested, installed, expected):
    assert missing_models(requested, installed) == expected


def test_cli_reports_benchmark_errors(tmp_path, capsys):
    exit_code = main(
        [
            "--questions",
            str(tmp_path / "missing.json"),
            "--papers",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1
    assert "Questions file not found" in capsys.readouterr().out
