"""Unit tests for benchmark loading and validation."""

import json

import pytest

from evaluation.benchmark import BenchmarkError, load_benchmark


@pytest.fixture
def papers(tmp_path):
    folder = tmp_path / "papers"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(b"%PDF")
    (folder / "b.pdf").write_bytes(b"%PDF")
    return folder


def _write(tmp_path, questions):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    return path


GOOD = {
    "id": "q1",
    "question": "What was the dataset?",
    "relevant": [{"paper": "a.pdf", "pages": [2, 3]}, {"paper": "b.pdf", "pages": [1]}],
}


def test_loads_valid_benchmark(tmp_path, papers):
    unanswerable = {"id": "q2", "question": "Capital of France?", "answerable": False}

    benchmark = load_benchmark(_write(tmp_path, [GOOD, unanswerable]), papers)

    assert [q.id for q in benchmark.questions] == ["q1", "q2"]
    assert [q.id for q in benchmark.answerable_questions] == ["q1"]
    assert benchmark.questions[0].relevant_pages == {
        ("a.pdf", 2),
        ("a.pdf", 3),
        ("b.pdf", 1),
    }
    assert benchmark.questions[1].relevant_pages == frozenset()
    assert [p.name for p in benchmark.paper_files()] == ["a.pdf", "b.pdf"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"id": ""}, "non-empty 'id'"),
        ({"question": "  "}, "no text"),
        ({"answerable": "yes"}, "true or false"),
        ({"relevant": []}, "lists no relevant pages"),
        ({"relevant": [{"paper": "a.pdf", "pages": []}]}, "non-empty 'pages'"),
        ({"relevant": [{"paper": "missing.pdf", "pages": [1]}]}, "not in"),
        ({"relevant": [{"paper": "a.pdf", "pages": [0]}]}, "starting at 1"),
        ({"relevant": [{"paper": "a.pdf", "pages": ["3"]}]}, "starting at 1"),
        ({"relevant": [{"paper": "a.pdf", "pages": [True]}]}, "starting at 1"),
    ],
)
def test_rejects_invalid_questions(tmp_path, papers, mutation, message):
    with pytest.raises(BenchmarkError, match=message):
        load_benchmark(_write(tmp_path, [{**GOOD, **mutation}]), papers)


def test_reads_optional_expected_answer(tmp_path, papers):
    with_answer = {**GOOD, "expected_answer": " The WMT 2014 dataset. "}

    benchmark = load_benchmark(
        _write(tmp_path, [with_answer, {**GOOD, "id": "q2"}]), papers
    )

    assert benchmark.questions[0].expected_answer == "The WMT 2014 dataset."
    assert benchmark.questions[1].expected_answer is None


def test_rejects_non_text_expected_answer(tmp_path, papers):
    with pytest.raises(BenchmarkError, match="'expected_answer' must be text"):
        load_benchmark(_write(tmp_path, [{**GOOD, "expected_answer": 42}]), papers)


def test_rejects_unanswerable_question_with_pages(tmp_path, papers):
    bad = {**GOOD, "answerable": False}

    with pytest.raises(BenchmarkError, match="unanswerable but lists"):
        load_benchmark(_write(tmp_path, [bad]), papers)


def test_rejects_duplicate_ids(tmp_path, papers):
    with pytest.raises(BenchmarkError, match="Duplicate question ids: q1"):
        load_benchmark(_write(tmp_path, [GOOD, GOOD]), papers)


def test_requires_an_answerable_question(tmp_path, papers):
    only_unanswerable = {"id": "q1", "question": "Anything?", "answerable": False}

    with pytest.raises(BenchmarkError, match="at least one answerable"):
        load_benchmark(_write(tmp_path, [only_unanswerable]), papers)


def test_requires_pdfs(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    questions = _write(
        tmp_path, [{**GOOD, "relevant": [{"paper": "a.pdf", "pages": [1]}]}]
    )

    with pytest.raises(BenchmarkError):
        load_benchmark(questions, empty)


def test_reports_missing_files_and_bad_json(tmp_path, papers):
    with pytest.raises(BenchmarkError, match="not found"):
        load_benchmark(tmp_path / "nope.json", papers)

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="not valid JSON"):
        load_benchmark(broken, papers)

    with pytest.raises(BenchmarkError, match="Papers folder not found"):
        load_benchmark(_write(tmp_path, [GOOD]), tmp_path / "no_such_folder")


def test_rejects_questions_file_without_questions(tmp_path, papers):
    with pytest.raises(BenchmarkError, match="non-empty 'questions'"):
        load_benchmark(_write(tmp_path, []), papers)
