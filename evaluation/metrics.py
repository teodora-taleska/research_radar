"""Retrieval evaluation metrics.

Relevance is judged per page. A retrieved passage counts as relevant when
its (paper, page) is among the labelled relevant pages of the question.
This keeps labels valid when chunking settings change.
"""

import math
import statistics
from collections.abc import Collection, Sequence

Page = tuple[str, int]


def precision_at_k(
    retrieved: Sequence[Page],
    relevant: Collection[Page],
    k: int,
) -> float:
    """Fraction of the top-k retrieved passages that are relevant.

    The denominator is always k, so returning fewer than k passages
    lowers precision.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    relevant_set = set(relevant)
    hits = sum(1 for page in retrieved[:k] if page in relevant_set)
    return hits / k


def recall_at_k(
    retrieved: Sequence[Page],
    relevant: Collection[Page],
    k: int,
) -> float:
    """Fraction of the relevant pages found in the top-k passages."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("At least one relevant page is required.")

    found = relevant_set & set(retrieved[:k])
    return len(found) / len(relevant_set)


def summarize_values(values: Sequence[float]) -> dict[str, float]:
    """Return mean, median, 95th percentile (nearest rank), min and max."""
    if not values:
        raise ValueError("At least one value is required.")

    ordered = sorted(values)
    p95_index = math.ceil(0.95 * len(ordered)) - 1

    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def summarize_latencies(latencies: Sequence[float]) -> dict[str, float]:
    """Summarise latency measurements in whatever unit they were given."""
    return summarize_values(latencies)
