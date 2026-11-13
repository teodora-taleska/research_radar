"""Unit tests for retrieval metrics."""

import pytest

from evaluation.metrics import precision_at_k, recall_at_k, summarize_latencies

RETRIEVED = [("a.pdf", 1), ("a.pdf", 2), ("b.pdf", 1), ("a.pdf", 3)]
RELEVANT = {("a.pdf", 1), ("a.pdf", 3), ("b.pdf", 9)}


@pytest.mark.parametrize(
    "k, expected",
    [(1, 1.0), (2, 0.5), (3, 1 / 3), (4, 0.5)],
)
def test_precision_at_k(k, expected):
    assert precision_at_k(RETRIEVED, RELEVANT, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    "k, expected",
    [(1, 1 / 3), (3, 1 / 3), (4, 2 / 3)],
)
def test_recall_at_k(k, expected):
    assert recall_at_k(RETRIEVED, RELEVANT, k) == pytest.approx(expected)


def test_precision_penalises_returning_fewer_than_k():
    assert precision_at_k([("a.pdf", 1), ("a.pdf", 3)], RELEVANT, 5) == pytest.approx(
        0.4
    )


def test_duplicate_pages_count_once_for_recall_but_each_for_precision():
    retrieved = [("a.pdf", 1), ("a.pdf", 1)]
    relevant = {("a.pdf", 1), ("a.pdf", 2)}

    assert precision_at_k(retrieved, relevant, 2) == 1.0
    assert recall_at_k(retrieved, relevant, 2) == 0.5


def test_nothing_retrieved_scores_zero():
    assert precision_at_k([], RELEVANT, 3) == 0.0
    assert recall_at_k([], RELEVANT, 3) == 0.0


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k])
def test_k_must_be_positive(metric):
    with pytest.raises(ValueError):
        metric(RETRIEVED, RELEVANT, 0)


def test_recall_requires_relevant_pages():
    with pytest.raises(ValueError):
        recall_at_k(RETRIEVED, set(), 3)


def test_latency_summary():
    summary = summarize_latencies([40.0, 10.0, 30.0, 20.0])

    assert summary == {
        "mean": 25.0,
        "median": 25.0,
        "p95": 40.0,
        "min": 10.0,
        "max": 40.0,
    }


def test_p95_uses_nearest_rank():
    assert summarize_latencies(list(range(1, 21)))["p95"] == 19


def test_latency_summary_requires_values():
    with pytest.raises(ValueError):
        summarize_latencies([])
