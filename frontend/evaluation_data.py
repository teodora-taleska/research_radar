"""Loading and shaping evaluation results for the Evaluation page.

The evaluation scripts write JSON reports to evaluation/results/. This module
reads them and prepares plain rows for the page. It does not import the
evaluation package, so the frontend only depends on the report files.
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_PREFIX = "retrieval"
LLM_PREFIX = "llm_comparison"
SCORES_PREFIX = "faithfulness"
SCORE_VALUES = (1, 2, 3, 4, 5)


class ResultsError(RuntimeError):
    """Raised when a results file cannot be read."""


def results_dir() -> Path:
    """Return the folder holding evaluation reports."""
    configured = os.getenv("RESEARCH_RADAR_RESULTS_DIR")
    return Path(configured) if configured else PROJECT_ROOT / "evaluation" / "results"


def list_runs(prefix: str, directory: Path | None = None) -> list[Path]:
    """Return report files with a prefix, newest first."""
    folder = directory or results_dir()
    if not folder.is_dir():
        return []
    return sorted(folder.glob(f"{prefix}_*.json"), reverse=True)


def load_run(path: Path) -> dict:
    """Read a report file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResultsError(f"Cannot read {Path(path).name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ResultsError(f"{Path(path).name} is not a report.")
    return data


def _when(report: dict) -> str:
    return str(report.get("created_at", ""))[:16].replace("T", " ")


# --- retrieval ---------------------------------------------------------------


def retrieval_label(report: dict) -> str:
    """Return a short description of a retrieval run for a selector."""
    config = report["config"]
    return (
        f"{_when(report)} | {config['embedding_model']} | "
        f"chunks {config['chunk_size']}/{config['chunk_overlap']} | "
        f"{config['questions_evaluated']} questions"
    )


def available_ks(report: dict) -> list[int]:
    """Return the K values that were evaluated."""
    return sorted(int(k) for k in report["summary"]["precision_at_k"])


def metrics_by_k(report: dict) -> list[dict]:
    """Return Precision@K and Recall@K for every evaluated K."""
    summary = report["summary"]
    return [
        {
            "K": k,
            "Precision@K": summary["precision_at_k"][str(k)],
            "Recall@K": summary["recall_at_k"][str(k)],
        }
        for k in available_ks(report)
    ]


def question_rows(report: dict, k: int) -> list[dict]:
    """Return one row per question for the chosen K."""
    return [
        {
            "ID": q["id"],
            "Question": q["question"],
            "Relevant pages": len(q["relevant"]),
            "Precision@K": q["precision_at_k"][str(k)],
            "Recall@K": q["recall_at_k"][str(k)],
            "Latency (ms)": q["latency_ms"],
        }
        for q in report["questions"]
    ]


def failure_cases(report: dict, k: int) -> list[dict]:
    """Return questions that missed relevant pages in the top K, worst first."""
    missed = [q for q in report["questions"] if q["recall_at_k"][str(k)] < 1.0]
    return sorted(missed, key=lambda q: (q["recall_at_k"][str(k)], q["id"]))


def missed_pages(question: dict, k: int) -> list[dict]:
    """Return the labelled pages that were not among the top K results."""
    found = {(r["paper"], r["page"]) for r in question["retrieved"][:k]}
    return [
        page
        for page in question["relevant"]
        if (page["paper"], page["page"]) not in found
    ]


def runs_comparison_rows(reports: list[dict], k: int) -> list[dict]:
    """Return one row per run that evaluated K, to compare configurations."""
    rows = []
    for report in reports:
        summary = report["summary"]
        if str(k) not in summary["precision_at_k"]:
            continue
        config = report["config"]
        rows.append(
            {
                "Run": _when(report),
                "Embedding model": config["embedding_model"],
                "Chunk size": config["chunk_size"],
                "Overlap": config["chunk_overlap"],
                "Precision@K": summary["precision_at_k"][str(k)],
                "Recall@K": summary["recall_at_k"][str(k)],
                "Median latency (ms)": summary["latency_ms"]["median"],
            }
        )
    return rows


# --- LLM comparison ----------------------------------------------------------


def llm_label(report: dict) -> str:
    """Return a short description of an LLM comparison run for a selector."""
    config = report["config"]
    return (
        f"{_when(report)} | {' vs '.join(config['models'])} | "
        f"{config['questions_answerable'] + config['questions_unanswerable']} questions"
    )


def scores_path(run_path: Path) -> Path:
    """Return the file holding manual faithfulness scores for a run."""
    name = Path(run_path).name
    if not name.startswith(f"{LLM_PREFIX}_"):
        raise ResultsError(f"{name} is not an LLM comparison report.")
    return Path(run_path).with_name(name.replace(LLM_PREFIX, SCORES_PREFIX, 1))


def _score_key(model: str, question_id: str) -> str:
    return f"{model}|{question_id}"


def load_scores(run_path: Path) -> dict[str, int]:
    """Return saved faithfulness scores keyed by 'model|question id'."""
    path = scores_path(run_path)
    if not path.is_file():
        return {}
    return {key: int(value) for key, value in load_run(path).items()}


def get_score(scores: dict[str, int], model: str, question_id: str) -> int | None:
    """Return one saved score, or None if the answer is not scored yet."""
    return scores.get(_score_key(model, question_id))


def save_score(
    run_path: Path,
    model: str,
    question_id: str,
    score: int | None,
) -> None:
    """Save or clear one faithfulness score (1 to 5)."""
    if score is not None and score not in SCORE_VALUES:
        raise ValueError("Score must be between 1 and 5.")

    scores = load_scores(run_path)
    key = _score_key(model, question_id)
    if score is None:
        scores.pop(key, None)
    else:
        scores[key] = score

    path = scores_path(run_path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def mean_faithfulness(scores: dict[str, int], model: str) -> tuple[float | None, int]:
    """Return a model's mean faithfulness score and how many answers were scored."""
    values = [v for key, v in scores.items() if key.split("|", 1)[0] == model]
    return (sum(values) / len(values) if values else None), len(values)


def model_summary_rows(report: dict, scores: dict[str, int]) -> list[dict]:
    """Return one headline-metrics row per model."""
    rows = []
    for model, data in report["models"].items():
        summary = data["summary"]
        latency = summary["latency_s"] or {}
        mean, scored = mean_faithfulness(scores, model)
        rows.append(
            {
                "Model": model,
                "Answered (answerable)": summary["answer_rate"],
                "Correct refusals": summary["correct_refusal_rate"],
                "Citation on relevant page": summary["citation_hit_rate"],
                "Citation precision": summary["citation_precision"],
                "Median latency (s)": latency.get("median"),
                "p95 latency (s)": latency.get("p95"),
                "Faithfulness (1-5)": mean,
                "Scored answers": scored,
                "Errors": summary["errors"],
            }
        )
    return rows


def question_comparison(report: dict) -> list[dict]:
    """Return each question with every model's answer side by side."""
    models = report["models"]
    first = next(iter(models.values()))["answers"]
    by_model = {
        name: {a["id"]: a for a in data["answers"]} for name, data in models.items()
    }
    return [
        {
            "id": a["id"],
            "question": a["question"],
            "answerable": a["answerable"],
            "expected_answer": a["expected_answer"],
            "answers": {name: answers[a["id"]] for name, answers in by_model.items()},
        }
        for a in first
    ]
