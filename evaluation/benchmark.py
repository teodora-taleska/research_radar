"""Loading and validation of the retrieval benchmark.

A benchmark is a folder of PDF papers plus a JSON file of questions. Each
answerable question lists the paper pages that contain its answer:

    {
      "questions": [
        {
          "id": "q01",
          "question": "What dataset was used for evaluation?",
          "answerable": true,
          "relevant": [{"paper": "paper_a.pdf", "pages": [4, 5]}]
        },
        {
          "id": "q02",
          "question": "Who won the 2018 football World Cup?",
          "answerable": false
        }
      ]
    }

Page numbers are one-based PDF page indexes (the page counter of a PDF
viewer), not the page numbers printed on the paper.
"""

import json
from dataclasses import dataclass
from pathlib import Path

Page = tuple[str, int]


class BenchmarkError(ValueError):
    """Raised when the benchmark files are missing or malformed."""


@dataclass(frozen=True)
class BenchmarkQuestion:
    """A benchmark question with its labelled relevant pages."""

    id: str
    question: str
    answerable: bool
    relevant_pages: frozenset[Page]


@dataclass(frozen=True)
class Benchmark:
    """A set of papers and labelled questions."""

    papers_dir: Path
    questions: list[BenchmarkQuestion]

    @property
    def answerable_questions(self) -> list[BenchmarkQuestion]:
        """Questions that have labelled relevant pages."""
        return [q for q in self.questions if q.answerable]

    def paper_files(self) -> list[Path]:
        """All PDF papers in the benchmark, sorted by name."""
        return sorted(self.papers_dir.glob("*.pdf"))


def _parse_question(raw: object, papers_dir: Path) -> BenchmarkQuestion:
    if not isinstance(raw, dict):
        raise BenchmarkError("Every question must be a JSON object.")

    question_id = raw.get("id")
    text = raw.get("question")
    if not isinstance(question_id, str) or not question_id.strip():
        raise BenchmarkError("Every question needs a non-empty 'id'.")
    if not isinstance(text, str) or not text.strip():
        raise BenchmarkError(f"Question '{question_id}' has no text.")

    answerable = raw.get("answerable", True)
    if not isinstance(answerable, bool):
        raise BenchmarkError(
            f"Question '{question_id}': 'answerable' must be true or false."
        )

    relevant = raw.get("relevant") or []

    if not answerable:
        if relevant:
            raise BenchmarkError(
                f"Question '{question_id}' is unanswerable but lists relevant pages."
            )
        return BenchmarkQuestion(question_id, text.strip(), False, frozenset())

    if not isinstance(relevant, list) or not relevant:
        raise BenchmarkError(
            f"Question '{question_id}' is answerable but lists no relevant pages."
        )

    pages: set[Page] = set()
    for entry in relevant:
        paper = entry.get("paper") if isinstance(entry, dict) else None
        entry_pages = entry.get("pages") if isinstance(entry, dict) else None
        if (
            not isinstance(paper, str)
            or not isinstance(entry_pages, list)
            or not entry_pages
        ):
            raise BenchmarkError(
                f"Question '{question_id}': each relevant entry needs "
                "'paper' and a non-empty 'pages' list."
            )
        if not (papers_dir / paper).is_file():
            raise BenchmarkError(
                f"Question '{question_id}' refers to '{paper}', which is "
                f"not in {papers_dir}."
            )
        for page in entry_pages:
            if isinstance(page, bool) or not isinstance(page, int) or page < 1:
                raise BenchmarkError(
                    f"Question '{question_id}': page numbers must be "
                    f"integers starting at 1, got {page!r}."
                )
            pages.add((paper, page))

    return BenchmarkQuestion(question_id, text.strip(), True, frozenset(pages))


def load_benchmark(questions_path: Path, papers_dir: Path) -> Benchmark:
    """Load and validate a benchmark.

    Raises:
        BenchmarkError: If files are missing, malformed or inconsistent.
    """
    questions_path = Path(questions_path)
    papers_dir = Path(papers_dir)

    if not questions_path.is_file():
        raise BenchmarkError(f"Questions file not found: {questions_path}")
    if not papers_dir.is_dir():
        raise BenchmarkError(f"Papers folder not found: {papers_dir}")

    try:
        data = json.loads(questions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{questions_path} is not valid JSON: {exc}") from exc

    raw_questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(raw_questions, list) or not raw_questions:
        raise BenchmarkError("The file must contain a non-empty 'questions' list.")

    questions = [_parse_question(raw, papers_dir) for raw in raw_questions]

    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise BenchmarkError(f"Duplicate question ids: {', '.join(duplicates)}")

    benchmark = Benchmark(papers_dir=papers_dir, questions=questions)

    if not benchmark.answerable_questions:
        raise BenchmarkError("The benchmark needs at least one answerable question.")
    if not benchmark.paper_files():
        raise BenchmarkError(f"No PDF files found in {papers_dir}.")

    return benchmark
