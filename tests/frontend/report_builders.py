"""Builds evaluation reports with the real evaluation code for the frontend tests."""

import json
from pathlib import Path

from app.generation.answer_generator import AnswerGenerator
from app.generation.prompts import INSUFFICIENT_EVIDENCE_MARKER
from app.vectorstore.chroma_store import StoredChunk
from evaluation.benchmark import Benchmark, BenchmarkQuestion
from evaluation.experiments.llm_comparison import run_comparison
from evaluation.run_evaluation import evaluate_retrieval

QUESTIONS = [
    BenchmarkQuestion(
        "q1",
        "What did paper A find?",
        True,
        frozenset({("a.pdf", 1), ("a.pdf", 2)}),
        "A found X.",
    ),
    BenchmarkQuestion(
        "q2",
        "What did paper B find?",
        True,
        frozenset({("b.pdf", 3)}),
        "B found Y.",
    ),
    BenchmarkQuestion("q3", "Capital of France?", False, frozenset()),
]
BENCHMARK = Benchmark(papers_dir=Path("."), questions=QUESTIONS)


def _chunk(paper, page, distance):
    return StoredChunk("id", "doc", paper, page, 0, f"{paper} page {page}", distance)


def _retrieve(question, top_k):
    if "paper A" in question:
        return [
            _chunk("a.pdf", 2, 0.2),
            _chunk("a.pdf", 5, 0.3),
            _chunk("a.pdf", 1, 0.4),
        ][:top_k]
    if "paper B" in question:
        return [
            _chunk("a.pdf", 4, 0.5),
            _chunk("a.pdf", 6, 0.6),
            _chunk("a.pdf", 7, 0.7),
        ][:top_k]
    return [_chunk("a.pdf", 1, 0.9)][:top_k]


class _Scripted:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt, system=None):
        if "France" in prompt:
            return (
                INSUFFICIENT_EVIDENCE_MARKER if self.reply != "always" else "Paris [1]."
            )
        return "Found it [1]."


def write_retrieval_report(
    folder, name="retrieval_20261113_120000.json", chunk_size=1200
):
    report = evaluate_retrieval(BENCHMARK, _retrieve, ks=[1, 3, 5])
    report = {
        "created_at": "2026-11-13T12:00:00",
        "config": {
            "embedding_model": "embeddinggemma",
            "embedding_dimension": 768,
            "distance_metric": "cosine",
            "chunk_size": chunk_size,
            "chunk_overlap": 200,
            "ks": [1, 3, 5],
            "papers": ["a.pdf", "b.pdf"],
            "indexed_chunks": 10,
            "questions_evaluated": 2,
            "questions_unanswerable": 1,
        },
        **report,
    }
    path = folder / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def write_llm_report(folder, name="llm_comparison_20261113_130000.json"):
    result = run_comparison(
        BENCHMARK,
        _retrieve,
        {
            "model-a": AnswerGenerator(_Scripted("refuses")),
            "model-b": AnswerGenerator(_Scripted("always")),
        },
        top_k=3,
    )
    report = {
        "created_at": "2026-11-13T13:00:00",
        "config": {
            "models": ["model-a", "model-b"],
            "embedding_model": "embeddinggemma",
            "chunk_size": 1200,
            "chunk_overlap": 200,
            "papers": ["a.pdf", "b.pdf"],
            "indexed_chunks": 10,
            "questions_answerable": 2,
            "questions_unanswerable": 1,
        },
        **result,
    }
    path = folder / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path
