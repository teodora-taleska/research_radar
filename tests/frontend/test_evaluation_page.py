"""Streamlit tests for the Evaluation page, using reports from the real runners."""

import json
from pathlib import Path

from evaluation_data import load_scores, scores_path
from report_builders import write_llm_report, write_retrieval_report
from streamlit.testing.v1 import AppTest

PAGE = str(
    Path(__file__).resolve().parents[2] / "frontend" / "pages" / "4_Evaluation.py"
)


def _run():
    return AppTest.from_file(PAGE, default_timeout=30).run()


def _select(at, label):
    return next(s for s in at.selectbox if s.label == label)


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_without_results_it_explains_how_to_create_them(results_dir):
    at = _run()

    assert not at.exception
    texts = [i.value for i in at.info]
    assert any("No retrieval results yet" in t for t in texts)
    assert any("No LLM comparison yet" in t for t in texts)
    assert any("evaluation.run_evaluation" in c.value for c in at.code)
    assert any("llm_comparison" in c.value for c in at.code)


def test_retrieval_tab_shows_metrics_for_the_default_k(results_dir):
    write_retrieval_report(results_dir)

    at = _run()

    assert not at.exception
    metrics = _metrics(at)
    assert metrics["Precision@5"] == "0.200"
    assert metrics["Recall@5"] == "0.500"
    assert metrics["Median latency"].endswith("ms")
    assert any("Precision and recall by K" in s.value for s in at.subheader)


def test_changing_k_updates_the_metrics_and_failure_cases(results_dir):
    write_retrieval_report(results_dir)
    at = _run()

    at.radio[0].set_value(3).run()

    assert _metrics(at)["Recall@3"] == "0.500"
    labels = [e.label for e in at.expander]
    assert any(label.startswith("q2 | recall 0.00") for label in labels)
    assert not any(label.startswith("q1 |") for label in labels)


def test_failure_case_lists_missed_pages(results_dir):
    write_retrieval_report(results_dir)
    at = _run()

    at.radio[0].set_value(1).run()

    texts = [m.value for m in at.markdown]
    assert any("Missed:** a.pdf page 1" in t for t in texts)
    assert any("Missed:** b.pdf page 3" in t for t in texts)


def test_runs_are_compared_when_there_are_several(results_dir):
    write_retrieval_report(results_dir, "retrieval_20261113_100000.json")
    write_retrieval_report(
        results_dir, "retrieval_20261113_110000.json", chunk_size=800
    )

    at = _run()

    assert any("Compare runs at K=5" in s.value for s in at.subheader)


def test_unreadable_report_is_reported_but_others_still_load(results_dir):
    write_retrieval_report(results_dir)
    (results_dir / "retrieval_20261113_999999.json").write_text(
        "{broken", encoding="utf-8"
    )

    at = _run()

    assert any(
        "Cannot read retrieval_20261113_999999.json" in w.value for w in at.warning
    )
    assert "Precision@5" in _metrics(at)


def test_llm_tab_summarises_models_and_compares_answers(results_dir):
    write_llm_report(results_dir)

    at = _run()

    assert not at.exception
    summary = next(d for d in at.dataframe if "Model" in d.value.columns)
    assert list(summary.value["Model"]) == ["model-a", "model-b"]
    assert any("Compare the answers" in s.value for s in at.subheader)
    assert any("Expected answer: A found X." in i.value for i in at.info)


def test_unanswerable_question_is_explained(results_dir):
    write_llm_report(results_dir)
    at = _run()

    question = _select(at, "Question")
    question.select(2).run()

    assert any("Unanswerable" in i.value for i in at.info)
    assert not at.exception


def test_scoring_an_answer_saves_it_and_updates_the_summary(results_dir):
    path = write_llm_report(results_dir)
    at = _run()
    key = f"score_{path.name}_model-a_q1"

    at.selectbox(key=key).select(4).run()

    assert load_scores(path) == {"model-a|q1": 4}
    summary = next(d for d in at.dataframe if "Model" in d.value.columns)
    row = summary.value.set_index("Model").loc["model-a"]
    assert row["Faithfulness (1-5)"] == 4.0
    assert row["Scored answers"] == 1

    at.selectbox(key=key).select("Not scored").run()

    assert load_scores(path) == {}
    assert json.loads(scores_path(path).read_text()) == {}


def test_page_renders_both_report_kinds_together(results_dir):
    write_retrieval_report(results_dir)
    write_llm_report(results_dir)

    at = _run()

    assert not at.exception
    assert [t.label for t in at.tabs] == ["Retrieval", "LLM comparison"]
