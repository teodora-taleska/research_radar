"""LLM comparison: answer the benchmark questions with several models.

Run from the project root with Ollama running:

    python -m evaluation.experiments.llm_comparison

Retrieval is identical for every model (the same passages are retrieved once
per question), so differences come from the language model alone. For each
model the script records speed, whether it answered or refused, and whether
its citations point to the labelled relevant pages. Faithfulness (is every
claim supported by the cited text) cannot be computed reliably, so it is
scored by hand on the Evaluation page.
"""

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

import ollama

from app.embeddings.provider import OllamaEmbeddingProvider
from app.generation.answer_generator import AnswerGenerator
from app.generation.llm_provider import LLMProviderError, OllamaLLMProvider
from app.retrieval.retriever import Retriever
from app.vectorstore.chroma_store import StoredChunk
from evaluation.benchmark import Benchmark, BenchmarkError, load_benchmark
from evaluation.metrics import summarize_values
from evaluation.run_evaluation import indexed_benchmark

DEFAULT_MODELS = ("qwen3:4b", "qwen3:4b-instruct")

Retrieve = Callable[[str, int], list[StoredChunk]]


def _answer_question(question, chunks: list[StoredChunk], generator: AnswerGenerator):
    """Generate one answer and describe how it behaved."""
    start = time.perf_counter()
    error = None
    result = None
    try:
        result = generator.generate(question.question, chunks)
    except LLMProviderError as exc:
        error = str(exc)
    latency_s = time.perf_counter() - start

    citations = []
    if result is not None:
        citations = [
            {
                "reference": citation.reference,
                "paper": citation.filename,
                "page": citation.page_number,
                "relevant": (citation.filename, citation.page_number)
                in question.relevant_pages,
            }
            for citation in result.citations
        ]

    return {
        "id": question.id,
        "question": question.question,
        "answerable": question.answerable,
        "expected_answer": question.expected_answer,
        "answer": result.answer if result else None,
        "answered": result.sufficient_evidence if result else None,
        "citations": citations,
        "latency_s": latency_s,
        "top_distance": chunks[0].distance if chunks else None,
        "error": error,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_model(answers: Sequence[dict]) -> dict:
    """Aggregate one model's answers into the headline metrics.

    "Refused" means the answer was reported as insufficient evidence. That
    includes answers rejected because they cited no retrieved passage.
    Questions that failed with an LLM error are counted separately and left
    out of every rate.
    """
    ok = [a for a in answers if a["error"] is None]
    answerable = [a for a in ok if a["answerable"]]
    unanswerable = [a for a in ok if not a["answerable"]]
    answered = [a for a in answerable if a["answered"]]

    all_citations = [c for a in answered for c in a["citations"]]
    answers_with_relevant_citation = [
        a for a in answered if any(c["relevant"] for c in a["citations"])
    ]

    return {
        "questions_answerable": len(answerable),
        "questions_unanswerable": len(unanswerable),
        "errors": len(answers) - len(ok),
        "answer_rate": _rate(len(answered), len(answerable)),
        "correct_refusal_rate": _rate(
            sum(1 for a in unanswerable if not a["answered"]), len(unanswerable)
        ),
        "citation_hit_rate": _rate(len(answers_with_relevant_citation), len(answered)),
        "citation_precision": _rate(
            sum(1 for c in all_citations if c["relevant"]), len(all_citations)
        ),
        "latency_s": summarize_values([a["latency_s"] for a in ok]) if ok else None,
    }


def run_comparison(
    benchmark: Benchmark,
    retrieve: Retrieve,
    generators: Mapping[str, AnswerGenerator],
    top_k: int = 5,
    prepare: Callable[[str], None] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Answer every benchmark question with every model.

    Args:
        benchmark: The labelled questions, answerable and unanswerable.
        retrieve: Function taking (question, top_k) and returning chunks.
        generators: Answer generators by model name.
        top_k: Passages given to each model.
        prepare: Called with the model name before its questions, for
            example to load the model so loading is not timed.
        progress: Called with (model, questions done, total) after each answer.
    """
    retrieved = {
        question.id: retrieve(question.question, top_k)
        for question in benchmark.questions
    }

    def distances(answerable: bool) -> dict | None:
        values = [
            retrieved[q.id][0].distance
            for q in benchmark.questions
            if q.answerable == answerable and retrieved[q.id]
        ]
        return summarize_values(values) if values else None

    models = {}
    total = len(benchmark.questions)
    for name, generator in generators.items():
        if prepare:
            prepare(name)
        answers = []
        for done, question in enumerate(benchmark.questions, start=1):
            answers.append(
                _answer_question(question, retrieved[question.id], generator)
            )
            if progress:
                progress(name, done, total)
        models[name] = {"summary": summarize_model(answers), "answers": answers}

    return {
        "top_k": top_k,
        "retrieval_top_distance": {
            "answerable": distances(True),
            "unanswerable": distances(False),
        },
        "models": models,
    }


def missing_models(requested: Sequence[str], installed: Sequence[str]) -> list[str]:
    """Return requested model names that are not installed in Ollama."""

    def normalise(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    available = {normalise(name) for name in installed}
    return [name for name in requested if normalise(name) not in available]


def _installed_models() -> list[str]:
    response = ollama.Client(host=os.getenv("OLLAMA_HOST")).list()
    return [model.model for model in response.models]


def _print_summary(report: dict, output_path: Path) -> None:
    print("\nLLM comparison")
    header = (
        f"{'model':<22}{'answered':>9}{'refused*':>9}{'cite hit':>9}"
        f"{'median s':>9}{'p95 s':>7}"
    )
    print(header)
    for name, data in report["models"].items():
        s = data["summary"]

        def pct(value):
            return "n/a" if value is None else f"{value:.0%}"

        latency = s["latency_s"]
        median = f"{latency['median']:.1f}" if latency else "n/a"
        p95 = f"{latency['p95']:.1f}" if latency else "n/a"
        print(
            f"{name:<22}{pct(s['answer_rate']):>9}{pct(s['correct_refusal_rate']):>9}"
            f"{pct(s['citation_hit_rate']):>9}{median:>9}{p95:>7}"
        )
    print("answered = answerable questions answered; refused* = unanswerable refused")
    print(f"\nSaved to {output_path}")
    print("Score faithfulness on the Evaluation page (LLM comparison tab).")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the LLM comparison from the command line."""
    parser = argparse.ArgumentParser(
        description="Compare language models on the benchmark."
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--questions", type=Path, default=Path("evaluation/datasets/questions.json")
    )
    parser.add_argument(
        "--papers", type=Path, default=Path("evaluation/datasets/papers")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        benchmark = load_benchmark(args.questions, args.papers)
    except BenchmarkError as exc:
        print(f"Benchmark error: {exc}")
        return 1

    try:
        absent = missing_models(args.models, _installed_models())
    except Exception as exc:
        print(f"Cannot reach Ollama: {exc}")
        return 1
    if absent:
        print("These models are not installed in Ollama: " + ", ".join(absent))
        for name in absent:
            print(f"  ollama pull {name}")
        return 1

    embedder = OllamaEmbeddingProvider()
    providers = {name: OllamaLLMProvider(model=name) for name in args.models}

    def prepare(name: str) -> None:
        print(f"\nLoading {name}...")
        try:
            providers[name].generate("Reply with the single word OK.")
        except LLMProviderError as exc:
            print(f"  warm-up failed: {exc}")

    def progress(name: str, done: int, total: int) -> None:
        print(f"  {name}: {done}/{total}", flush=True)

    print(f"Indexing {len(benchmark.paper_files())} papers...")
    try:
        with indexed_benchmark(
            benchmark, embedder, args.chunk_size, args.chunk_overlap
        ) as indexed:
            retriever = Retriever(embedder, indexed.store)
            result = run_comparison(
                benchmark,
                lambda question, top_k: retriever.retrieve(question, top_k=top_k),
                {name: AnswerGenerator(llm) for name, llm in providers.items()},
                top_k=args.top_k,
                prepare=prepare,
                progress=progress,
            )
    except BenchmarkError as exc:
        print(f"Benchmark error: {exc}")
        return 1

    now = datetime.now()
    report = {
        "created_at": now.isoformat(timespec="seconds"),
        "config": {
            "models": list(args.models),
            "embedding_model": embedder.model,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "papers": sorted(indexed.page_counts),
            "indexed_chunks": indexed.total_chunks,
            "questions_answerable": len(benchmark.answerable_questions),
            "questions_unanswerable": len(benchmark.questions)
            - len(benchmark.answerable_questions),
        },
        **result,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"llm_comparison_{now:%Y%m%d_%H%M%S}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_summary(report, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
