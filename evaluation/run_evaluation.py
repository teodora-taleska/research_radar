"""Retrieval evaluation: index the benchmark papers and measure retrieval.

Run from the project root with Ollama running:

    python -m evaluation.run_evaluation

The papers are indexed into a temporary store, so your own uploaded
documents are never touched. Results are written to evaluation/results/.
"""

import argparse
import json
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from app.database.connection import create_db_engine, create_session_factory
from app.database.repositories import DocumentRepository
from app.embeddings.provider import OllamaEmbeddingProvider
from app.ingestion.pipeline import process_pdf
from app.retrieval.retriever import Retriever
from app.services.document_service import DocumentService
from app.vectorstore.chroma_store import ChromaDocumentStore, StoredChunk
from evaluation.benchmark import Benchmark, BenchmarkError, load_benchmark
from evaluation.metrics import precision_at_k, recall_at_k, summarize_latencies

DEFAULT_KS = (1, 3, 5, 10)

Retrieve = Callable[[str, int], list[StoredChunk]]


def build_index(
    benchmark: Benchmark,
    service: DocumentService,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[dict[str, int], int]:
    """Index every benchmark paper.

    Returns:
        A mapping of paper name to page count, and the total chunk count.
    """
    page_counts: dict[str, int] = {}
    total_chunks = 0

    for pdf_path in benchmark.paper_files():
        document = process_pdf(
            pdf_path.read_bytes(),
            pdf_path.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        total_chunks += service.index_document(document)
        page_counts[pdf_path.name] = document.page_count

    return page_counts, total_chunks


def validate_page_labels(
    benchmark: Benchmark,
    page_counts: dict[str, int],
) -> None:
    """Reject labelled pages that do not exist in their paper."""
    for question in benchmark.answerable_questions:
        for paper, page in sorted(question.relevant_pages):
            count = page_counts.get(paper)
            if count is None or page > count:
                raise BenchmarkError(
                    f"Question '{question.id}' labels page {page} of "
                    f"'{paper}', but that paper has "
                    f"{count if count is not None else 'no indexed'} pages."
                )


def evaluate_retrieval(
    benchmark: Benchmark,
    retrieve: Retrieve,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict:
    """Measure Precision@K, Recall@K and latency on answerable questions.

    Args:
        benchmark: The labelled questions.
        retrieve: Function taking (question, top_k) and returning chunks.
        ks: Cut-off values to evaluate.

    Returns:
        A dict with a "summary" and one entry per question.
    """
    k_values = sorted(set(ks))
    if not k_values or k_values[0] < 1:
        raise ValueError("ks must contain positive integers.")
    max_k = k_values[-1]

    question_results = []
    latencies_ms: list[float] = []

    for question in benchmark.answerable_questions:
        start = time.perf_counter()
        chunks = retrieve(question.question, max_k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency_ms)

        pages = [(chunk.filename, chunk.page_number) for chunk in chunks]

        question_results.append(
            {
                "id": question.id,
                "question": question.question,
                "relevant": [
                    {"paper": paper, "page": page}
                    for paper, page in sorted(question.relevant_pages)
                ],
                "retrieved": [
                    {
                        "rank": rank,
                        "paper": chunk.filename,
                        "page": chunk.page_number,
                        "distance": chunk.distance,
                        "relevant": (chunk.filename, chunk.page_number)
                        in question.relevant_pages,
                    }
                    for rank, chunk in enumerate(chunks, start=1)
                ],
                "precision_at_k": {
                    str(k): precision_at_k(pages, question.relevant_pages, k)
                    for k in k_values
                },
                "recall_at_k": {
                    str(k): recall_at_k(pages, question.relevant_pages, k)
                    for k in k_values
                },
                "latency_ms": latency_ms,
            }
        )

    def mean_of(metric: str, k: int) -> float:
        values = [result[metric][str(k)] for result in question_results]
        return sum(values) / len(values)

    return {
        "summary": {
            "precision_at_k": {str(k): mean_of("precision_at_k", k) for k in k_values},
            "recall_at_k": {str(k): mean_of("recall_at_k", k) for k in k_values},
            "latency_ms": summarize_latencies(latencies_ms),
        },
        "questions": question_results,
    }


def _print_report(report: dict, output_path: Path) -> None:
    summary = report["summary"]
    print("\nRetrieval results")
    print(f"{'K':>4} {'Precision@K':>12} {'Recall@K':>10}")
    for k, precision in summary["precision_at_k"].items():
        print(f"{k:>4} {precision:>12.3f} {summary['recall_at_k'][k]:>10.3f}")

    latency = summary["latency_ms"]
    print(
        f"\nLatency (ms): mean {latency['mean']:.0f}, "
        f"median {latency['median']:.0f}, p95 {latency['p95']:.0f}"
    )

    max_k = list(summary["recall_at_k"])[-1]
    failures = [q["id"] for q in report["questions"] if q["recall_at_k"][max_k] == 0]
    if failures:
        print(
            f"Questions with no relevant page in the top {max_k}: {', '.join(failures)}"
        )

    print(f"\nSaved to {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the retrieval evaluation from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate Research Radar retrieval.")
    parser.add_argument(
        "--questions", type=Path, default=Path("evaluation/datasets/questions.json")
    )
    parser.add_argument(
        "--papers", type=Path, default=Path("evaluation/datasets/papers")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        benchmark = load_benchmark(args.questions, args.papers)
    except BenchmarkError as exc:
        print(f"Benchmark error: {exc}")
        return 1

    embedder = OllamaEmbeddingProvider()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work_dir:
        work = Path(work_dir)
        store = ChromaDocumentStore(persist_directory=work / "chroma")
        engine = create_db_engine(f"sqlite:///{(work / 'eval.db').as_posix()}")
        service = DocumentService(
            embedder, store, DocumentRepository(create_session_factory(engine))
        )

        print(f"Indexing {len(benchmark.paper_files())} papers...")
        try:
            page_counts, total_chunks = build_index(
                benchmark, service, args.chunk_size, args.chunk_overlap
            )
            validate_page_labels(benchmark, page_counts)
        except BenchmarkError as exc:
            print(f"Benchmark error: {exc}")
            engine.dispose()
            return 1

        retriever = Retriever(embedder, store)
        embedder.embed_text("warm-up")  # exclude model loading from latency

        print(f"Evaluating {len(benchmark.answerable_questions)} questions...")
        report = evaluate_retrieval(
            benchmark,
            lambda question, top_k: retriever.retrieve(question, top_k=top_k),
            ks=args.ks,
        )
        engine.dispose()

    now = datetime.now()
    report = {
        "created_at": now.isoformat(timespec="seconds"),
        "config": {
            "embedding_model": embedder.model,
            "embedding_dimension": embedder.embedding_dimension,
            "distance_metric": "cosine",
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "ks": sorted(set(args.ks)),
            "papers": sorted(page_counts),
            "indexed_chunks": total_chunks,
            "questions_evaluated": len(benchmark.answerable_questions),
            "questions_unanswerable": len(benchmark.questions)
            - len(benchmark.answerable_questions),
        },
        **report,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"retrieval_{now:%Y%m%d_%H%M%S}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_report(report, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
