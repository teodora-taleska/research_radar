"""Tests for the helpers that shape evaluation reports for the page."""

import json

import pytest
from evaluation_data import (
    ResultsError,
    available_ks,
    failure_cases,
    get_score,
    list_runs,
    load_run,
    load_scores,
    mean_faithfulness,
    metrics_by_k,
    missed_pages,
    model_summary_rows,
    question_comparison,
    question_rows,
    runs_comparison_rows,
    save_score,
    scores_path,
)
from evaluation_data import results_dir as get_results_dir
from report_builders import write_llm_report, write_retrieval_report


@pytest.fixture
def retrieval(results_dir):
    return load_run(write_retrieval_report(results_dir))


@pytest.fixture
def llm_path(results_dir):
    return write_llm_report(results_dir)


def test_results_dir_uses_the_environment_variable(results_dir):
    assert get_results_dir() == results_dir


def test_results_dir_defaults_to_the_project_results_folder(monkeypatch):
    monkeypatch.delenv("RESEARCH_RADAR_RESULTS_DIR", raising=False)

    assert get_results_dir().parts[-2:] == ("evaluation", "results")


def test_list_runs_returns_newest_first_and_only_that_kind(results_dir):
    write_retrieval_report(results_dir, "retrieval_20261113_100000.json")
    write_retrieval_report(results_dir, "retrieval_20261113_120000.json")
    write_llm_report(results_dir)

    names = [p.name for p in list_runs("retrieval")]

    assert names == [
        "retrieval_20261113_120000.json",
        "retrieval_20261113_100000.json",
    ]


def test_list_runs_on_a_missing_folder_is_empty(tmp_path):
    assert list_runs("retrieval", tmp_path / "nope") == []


def test_load_run_reports_unreadable_files(results_dir):
    broken = results_dir / "retrieval_1.json"
    broken.write_text("{nope", encoding="utf-8")
    listed = results_dir / "retrieval_2.json"
    listed.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(ResultsError, match="Cannot read retrieval_1.json"):
        load_run(broken)
    with pytest.raises(ResultsError, match="not a report"):
        load_run(listed)


def test_metrics_by_k(retrieval):
    rows = metrics_by_k(retrieval)

    assert available_ks(retrieval) == [1, 3, 5]
    assert [r["K"] for r in rows] == [1, 3, 5]
    # q1 finds a.pdf p2 first, q2 never finds its page.
    assert rows[0]["Precision@K"] == pytest.approx(0.5)
    assert rows[0]["Recall@K"] == pytest.approx(0.25)
    assert rows[2]["Recall@K"] == pytest.approx(0.5)


def test_question_rows(retrieval):
    rows = question_rows(retrieval, 3)

    assert [r["ID"] for r in rows] == ["q1", "q2"]
    assert rows[0]["Relevant pages"] == 2
    assert rows[0]["Recall@K"] == 1.0
    assert rows[1]["Recall@K"] == 0.0
    assert rows[0]["Latency (ms)"] >= 0


def test_failure_cases_are_worst_first_and_exclude_perfect_questions(retrieval):
    assert [q["id"] for q in failure_cases(retrieval, 1)] == ["q2", "q1"]
    assert [q["id"] for q in failure_cases(retrieval, 3)] == ["q2"]


def test_missed_pages_lists_what_was_not_retrieved(retrieval):
    q1 = next(q for q in retrieval["questions"] if q["id"] == "q1")

    assert missed_pages(q1, 1) == [{"paper": "a.pdf", "page": 1}]
    assert missed_pages(q1, 3) == []


def test_runs_comparison_rows_skip_runs_without_that_k(results_dir):
    first = load_run(
        write_retrieval_report(results_dir, "retrieval_20261113_100000.json")
    )
    second = load_run(
        write_retrieval_report(
            results_dir, "retrieval_20261113_110000.json", chunk_size=800
        )
    )
    second["summary"]["precision_at_k"].pop("5")

    rows = runs_comparison_rows([first, second], 5)

    assert len(rows) == 1
    assert rows[0]["Chunk size"] == 1200
    assert len(runs_comparison_rows([first, second], 3)) == 2


def test_scores_path_belongs_to_the_run(results_dir, llm_path):
    assert scores_path(llm_path).name == "faithfulness_20261113_130000.json"
    with pytest.raises(ResultsError):
        scores_path(results_dir / "retrieval_20261113_120000.json")


def test_scores_can_be_saved_changed_and_cleared(llm_path):
    assert load_scores(llm_path) == {}

    save_score(llm_path, "model-a", "q1", 4)
    save_score(llm_path, "model-b", "q1", 2)
    save_score(llm_path, "model-a", "q1", 5)

    scores = load_scores(llm_path)
    assert get_score(scores, "model-a", "q1") == 5
    assert get_score(scores, "model-b", "q2") is None
    assert json.loads(scores_path(llm_path).read_text()) == {
        "model-a|q1": 5,
        "model-b|q1": 2,
    }

    save_score(llm_path, "model-a", "q1", None)

    assert load_scores(llm_path) == {"model-b|q1": 2}


@pytest.mark.parametrize("score", [0, 6, 3.5])
def test_invalid_scores_are_rejected(llm_path, score):
    with pytest.raises(ValueError):
        save_score(llm_path, "model-a", "q1", score)


def test_mean_faithfulness_is_per_model():
    scores = {"model-a|q1": 4, "model-a|q2": 2, "model-b|q1": 5}

    assert mean_faithfulness(scores, "model-a") == (3.0, 2)
    assert mean_faithfulness(scores, "model-b") == (5.0, 1)
    assert mean_faithfulness(scores, "model-c") == (None, 0)


def test_model_summary_rows(llm_path):
    report = load_run(llm_path)
    rows = {r["Model"]: r for r in model_summary_rows(report, {"model-a|q1": 4})}

    assert list(rows) == ["model-a", "model-b"]
    assert rows["model-a"]["Answered (answerable)"] == 1.0
    assert rows["model-a"]["Correct refusals"] == 1.0
    assert rows["model-b"]["Correct refusals"] == 0.0
    assert rows["model-a"]["Faithfulness (1-5)"] == 4.0
    assert rows["model-a"]["Scored answers"] == 1
    assert rows["model-b"]["Faithfulness (1-5)"] is None
    assert rows["model-a"]["Median latency (s)"] >= 0


def test_question_comparison_puts_models_side_by_side(llm_path):
    items = question_comparison(load_run(llm_path))

    assert [i["id"] for i in items] == ["q1", "q2", "q3"]
    assert items[0]["expected_answer"] == "A found X."
    assert set(items[0]["answers"]) == {"model-a", "model-b"}
    assert items[2]["answerable"] is False
    assert items[2]["answers"]["model-a"]["answered"] is False
    assert items[2]["answers"]["model-b"]["answered"] is True
