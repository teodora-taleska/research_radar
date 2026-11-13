"""Tests for the retrieval evaluation runner."""

import json

import pymupdf
import pytest

from app.retrieval.retriever import Retriever
from app.services.document_service import DocumentService
from app.vectorstore.chroma_store import ChromaDocumentStore, StoredChunk
from evaluation.benchmark import BenchmarkError, load_benchmark
from evaluation.run_evaluation import (
    build_index,
    evaluate_retrieval,
    main,
    validate_page_labels,
)

VOCABULARY = ["attention", "dataset", "patients", "randomised"]


class KeywordEmbedder:
    """Embeds text as keyword counts so retrieval is predictable."""

    def embed_text(self, text):
        return self.embed_texts([text])[0]

    def embed_texts(self, texts):
        return [
            [float(text.lower().count(word)) for word in VOCABULARY] + [0.1]
            for text in texts
        ]


def _pdf(path, *page_texts):
    document = pymupdf.open()
    for text in page_texts:
        document.new_page().insert_text((72, 72), text)
    document.save(str(path))
    document.close()


@pytest.fixture
def benchmark_files(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    _pdf(papers / "nlp.pdf", "Self attention layers.", "The dataset was large.")
    _pdf(papers / "trial.pdf", "Patients were randomised.", "More patients enrolled.")

    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "question": "How does attention work?",
                        "relevant": [{"paper": "nlp.pdf", "pages": [1]}],
                    },
                    {
                        "id": "q2",
                        "question": "How were patients randomised?",
                        "relevant": [{"paper": "trial.pdf", "pages": [1]}],
                    },
                    {
                        "id": "q3",
                        "question": "Capital of France?",
                        "answerable": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return questions, papers


def _chunk(paper, page, distance=0.1):
    return StoredChunk("id", "doc", paper, page, 0, "text", distance)


def test_evaluates_with_a_scripted_retriever(benchmark_files):
    benchmark = load_benchmark(*benchmark_files)
    calls = []

    def retrieve(question, top_k):
        calls.append((question, top_k))
        if "attention" in question:
            return [_chunk("nlp.pdf", 1), _chunk("nlp.pdf", 2)]
        return [_chunk("nlp.pdf", 2), _chunk("trial.pdf", 1)]

    report = evaluate_retrieval(benchmark, retrieve, ks=[2, 1, 2])

    assert [c[1] for c in calls] == [2, 2]
    assert [c[0] for c in calls] == [
        "How does attention work?",
        "How were patients randomised?",
    ]
    summary = report["summary"]
    assert summary["precision_at_k"] == {"1": 0.5, "2": 0.5}
    assert summary["recall_at_k"] == {"1": 0.5, "2": 1.0}
    assert set(summary["latency_ms"]) == {"mean", "median", "p95", "min", "max"}

    second = report["questions"][1]
    assert [r["relevant"] for r in second["retrieved"]] == [False, True]
    assert second["recall_at_k"] == {"1": 0.0, "2": 1.0}
    assert second["relevant"] == [{"paper": "trial.pdf", "page": 1}]


def test_unanswerable_questions_are_not_evaluated(benchmark_files):
    benchmark = load_benchmark(*benchmark_files)

    report = evaluate_retrieval(benchmark, lambda q, k: [_chunk("nlp.pdf", 1)])

    assert [q["id"] for q in report["questions"]] == ["q1", "q2"]


def test_rejects_invalid_ks(benchmark_files):
    benchmark = load_benchmark(*benchmark_files)

    with pytest.raises(ValueError):
        evaluate_retrieval(benchmark, lambda q, k: [], ks=[0, 3])


def test_end_to_end_with_real_index_and_store(benchmark_files, tmp_path, repository):
    benchmark = load_benchmark(*benchmark_files)
    embedder = KeywordEmbedder()
    store = ChromaDocumentStore(persist_directory=tmp_path / "chroma")
    service = DocumentService(embedder, store, repository)

    page_counts, total_chunks = build_index(benchmark, service, 1200, 200)
    validate_page_labels(benchmark, page_counts)
    retriever = Retriever(embedder, store)
    report = evaluate_retrieval(
        benchmark,
        lambda question, top_k: retriever.retrieve(question, top_k=top_k),
        ks=[1, 3],
    )

    assert page_counts == {"nlp.pdf": 2, "trial.pdf": 2}
    assert total_chunks == 4
    assert report["summary"]["precision_at_k"]["1"] == 1.0
    assert report["summary"]["recall_at_k"]["1"] == 1.0
    assert report["questions"][0]["retrieved"][0]["distance"] >= 0.0


def test_page_labels_beyond_the_paper_are_rejected(benchmark_files):
    benchmark = load_benchmark(*benchmark_files)

    with pytest.raises(BenchmarkError, match="labels page 1 of 'nlp.pdf'"):
        validate_page_labels(benchmark, {"nlp.pdf": 0, "trial.pdf": 2})


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
