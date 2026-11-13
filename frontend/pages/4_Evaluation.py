"""Evaluation dashboard: retrieval metrics and LLM comparison."""

import altair as alt
import pandas as pd
import streamlit as st
from evaluation_data import (
    LLM_PREFIX,
    RETRIEVAL_PREFIX,
    SCORE_VALUES,
    ResultsError,
    available_ks,
    failure_cases,
    get_score,
    list_runs,
    llm_label,
    load_run,
    load_scores,
    metrics_by_k,
    missed_pages,
    model_summary_rows,
    question_comparison,
    question_rows,
    results_dir,
    retrieval_label,
    runs_comparison_rows,
    save_score,
)

st.set_page_config(page_title="Evaluation", layout="wide")

BLUE = "#2a78d6"
ORANGE = "#eb6834"
SERIES_COLORS = [BLUE, ORANGE, "#1baf7a", "#eda100"]
NOT_SCORED = "Not scored"
BEHAVIOUR_METRICS = (
    "Answered (answerable)",
    "Correct refusals",
    "Citation on relevant page",
)

RETRIEVAL_COMMAND = "python -m evaluation.run_evaluation"
LLM_COMMAND = "python -m evaluation.experiments.llm_comparison"


def load_reports(prefix: str) -> list[tuple]:
    """Load every readable report of a kind, newest first."""
    reports = []
    for path in list_runs(prefix):
        try:
            reports.append((path, load_run(path)))
        except ResultsError as exc:
            st.warning(str(exc))
    return reports


def save_score_from_widget(run_path, model: str, question_id: str, key: str) -> None:
    """Save the score chosen in a selectbox."""
    value = st.session_state[key]
    try:
        save_score(run_path, model, question_id, None if value == NOT_SCORED else value)
    except OSError as exc:
        st.toast(f"Could not save the score: {exc}")


def render_retrieval() -> None:
    reports = load_reports(RETRIEVAL_PREFIX)
    if not reports:
        st.info(
            "No retrieval results yet. Prepare the benchmark "
            "(evaluation/datasets/README.md) and run:"
        )
        st.code(RETRIEVAL_COMMAND, language="powershell")
        return

    choice = st.selectbox(
        "Run",
        options=range(len(reports)),
        format_func=lambda i: retrieval_label(reports[i][1]),
    )
    report = reports[choice][1]
    config = report["config"]

    ks = available_ks(report)
    k = st.radio(
        "Cut-off K: passages retrieved per question",
        ks,
        index=ks.index(5) if 5 in ks else len(ks) - 1,
        horizontal=True,
    )

    summary = report["summary"]
    latency = summary["latency_ms"]
    precision, recall, median, p95 = st.columns(4)
    precision.metric(
        f"Precision@{k}",
        f"{summary['precision_at_k'][str(k)]:.3f}",
        help="Share of the top K passages that come from a labelled relevant page.",
    )
    recall.metric(
        f"Recall@{k}",
        f"{summary['recall_at_k'][str(k)]:.3f}",
        help="Share of the labelled relevant pages found in the top K passages.",
    )
    median.metric("Median latency", f"{latency['median']:.0f} ms")
    p95.metric("95th percentile latency", f"{latency['p95']:.0f} ms")
    st.caption(
        f"{config['questions_evaluated']} answerable questions on "
        f"{len(config['papers'])} papers ({config['indexed_chunks']} indexed "
        f"passages). {config['embedding_model']}, {config['distance_metric']} "
        f"distance, chunk size {config['chunk_size']}, overlap "
        f"{config['chunk_overlap']}."
    )

    st.subheader("Precision and recall by K")
    by_k = pd.DataFrame(metrics_by_k(report))
    long_form = by_k.melt("K", var_name="Metric", value_name="Score")
    chart = (
        alt.Chart(long_form)
        .mark_line(point=alt.OverlayMarkDef(size=70, filled=True))
        .encode(
            x=alt.X("K:O", title="K (passages retrieved)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=["Precision@K", "Recall@K"], range=[BLUE, ORANGE]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["Metric", "K", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Recall rises with K because more passages can contain the relevant "
        "pages. Precision falls because most questions have only one or two "
        "relevant pages."
    )
    with st.expander("Show the values"):
        st.dataframe(by_k, hide_index=True, width="stretch")

    st.subheader(f"Results per question at K={k}")
    st.dataframe(
        pd.DataFrame(question_rows(report, k)),
        hide_index=True,
        width="stretch",
        column_config={
            "Precision@K": st.column_config.ProgressColumn(
                f"Precision@{k}", min_value=0, max_value=1, format="%.2f"
            ),
            "Recall@K": st.column_config.ProgressColumn(
                f"Recall@{k}", min_value=0, max_value=1, format="%.2f"
            ),
            "Latency (ms)": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    cases = failure_cases(report, k)
    st.subheader(f"Failure cases at K={k}")
    if not cases:
        st.success(f"Every labelled page was found in the top {k} for all questions.")
    else:
        st.caption(
            f"{len(cases)} of {len(report['questions'])} questions missed at "
            f"least one relevant page in the top {k}. Open one to see what was "
            "retrieved instead."
        )
    for case in cases:
        recall_value = case["recall_at_k"][str(k)]
        with st.expander(
            f"{case['id']} | recall {recall_value:.2f} | {case['question']}"
        ):
            missed = ", ".join(
                f"{page['paper']} page {page['page']}" for page in missed_pages(case, k)
            )
            st.markdown(f"**Missed:** {missed}")
            retrieved = pd.DataFrame(
                [
                    {
                        "Rank": r["rank"],
                        "Paper": r["paper"],
                        "Page": r["page"],
                        "Distance": r["distance"],
                        "Labelled relevant": "yes" if r["relevant"] else "no",
                    }
                    for r in case["retrieved"][:k]
                ]
            )
            st.dataframe(retrieved, hide_index=True, width="stretch")

    if len(reports) > 1:
        st.subheader(f"Compare runs at K={k}")
        rows = runs_comparison_rows([r for _, r in reports], k)
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Median latency (ms)": st.column_config.NumberColumn(format="%.0f"),
                "Precision@K": st.column_config.NumberColumn(
                    f"Precision@{k}", format="%.3f"
                ),
                "Recall@K": st.column_config.NumberColumn(f"Recall@{k}", format="%.3f"),
            },
        )


def render_answer_card(run_path, model: str, record: dict) -> None:
    """Show one model's answer to a question and its faithfulness score."""
    st.markdown(f"**{model}**")
    if record["error"]:
        st.error(record["error"])
        return

    behaviour = "Answered" if record["answered"] else "Refused"
    st.caption(f"{behaviour} | {record['latency_s']:.1f} s")

    if record["answered"]:
        st.markdown(record["answer"])
        for citation in record["citations"]:
            status = (
                "labelled relevant page"
                if citation["relevant"]
                else "not a labelled page"
            )
            st.caption(
                f"[{citation['reference']}] {citation['paper']}, page "
                f"{citation['page']} ({status})"
            )
    else:
        st.warning(record["answer"])

    if record["answerable"] and record["answered"]:
        key = f"score_{run_path.name}_{model}_{record['id']}"
        saved = get_score(load_scores(run_path), model, record["id"])
        options = [NOT_SCORED, *SCORE_VALUES]
        st.selectbox(
            "Faithfulness (1-5)",
            options,
            index=options.index(saved) if saved else 0,
            key=key,
            on_change=save_score_from_widget,
            args=(run_path, model, record["id"], key),
            help="Is every claim supported by the cited text? See the scoring guide.",
        )


def render_llm() -> None:
    reports = load_reports(LLM_PREFIX)
    if not reports:
        st.info(
            "No LLM comparison yet. It answers every benchmark question with "
            "each model and needs both models installed in Ollama:"
        )
        st.code(LLM_COMMAND, language="powershell")
        return

    choice = st.selectbox(
        "Run",
        options=range(len(reports)),
        format_func=lambda i: llm_label(reports[i][1]),
        key="llm_run",
    )
    run_path, report = reports[choice]
    config = report["config"]
    models = list(report["models"])
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(models))]
    scores = load_scores(run_path)

    st.caption(
        f"{config['questions_answerable']} answerable and "
        f"{config['questions_unanswerable']} unanswerable questions. Each model "
        f"received the same top {report['top_k']} passages, so differences come "
        "from the language model."
    )

    st.subheader("Summary")
    summary = pd.DataFrame(model_summary_rows(report, scores))
    percent = {"format": "percent", "min_value": 0, "max_value": 1}
    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            "Answered (answerable)": st.column_config.NumberColumn(
                help="Answerable questions that received a cited answer.", **percent
            ),
            "Correct refusals": st.column_config.NumberColumn(
                help="Unanswerable questions where the model declined to answer.",
                **percent,
            ),
            "Citation on relevant page": st.column_config.NumberColumn(
                help="Answers with at least one citation to a labelled relevant page.",
                **percent,
            ),
            "Citation precision": st.column_config.NumberColumn(
                help="Share of all citations that point to a labelled relevant page.",
                **percent,
            ),
            "Median latency (s)": st.column_config.NumberColumn(format="%.1f"),
            "p95 latency (s)": st.column_config.NumberColumn(format="%.1f"),
            "Faithfulness (1-5)": st.column_config.NumberColumn(
                help="Mean of your manual scores below.", format="%.2f"
            ),
        },
    )
    st.caption(
        "Citation checks are automatic and only test that the cited page is a "
        "labelled one. Faithfulness is your manual score of each answer."
    )

    speed, behaviour = st.columns([2, 3])
    color_scale = alt.Scale(domain=models, range=colors)
    with speed:
        st.markdown("**Median answer time (seconds)**")
        latency_rows = [
            {
                "Model": row["Model"],
                "Median": row["Median latency (s)"],
                "p95": row["p95 latency (s)"],
            }
            for row in summary.to_dict("records")
            if row["Median latency (s)"] is not None
        ]
        if latency_rows:
            latency_chart = (
                alt.Chart(pd.DataFrame(latency_rows))
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Median:Q", title=None),
                    color=alt.Color("Model:N", scale=color_scale, legend=None),
                    tooltip=[
                        "Model",
                        alt.Tooltip("Median:Q", format=".1f"),
                        alt.Tooltip("p95:Q", format=".1f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(latency_chart, width="stretch")
    with behaviour:
        st.markdown("**Answering behaviour**")
        behaviour_rows = [
            {"Metric": metric, "Model": row["Model"], "Rate": row[metric]}
            for row in summary.to_dict("records")
            for metric in BEHAVIOUR_METRICS
            if row[metric] is not None
        ]
        if behaviour_rows:
            behaviour_chart = (
                alt.Chart(pd.DataFrame(behaviour_rows))
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    y=alt.Y(
                        "Metric:N",
                        title=None,
                        sort=list(BEHAVIOUR_METRICS),
                        axis=alt.Axis(labelLimit=220),
                    ),
                    yOffset="Model:N",
                    x=alt.X(
                        "Rate:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        title=None,
                        axis=alt.Axis(format="%"),
                    ),
                    color=alt.Color(
                        "Model:N",
                        scale=color_scale,
                        legend=alt.Legend(orient="top", title=None),
                    ),
                    tooltip=["Model", "Metric", alt.Tooltip("Rate:Q", format=".0%")],
                )
                .properties(height=260)
            )
            st.altair_chart(behaviour_chart, width="stretch")

    distances = report["retrieval_top_distance"]
    st.subheader("Does retrieval tell answerable from unanswerable questions?")
    st.caption(
        "Distance of the best retrieved passage (lower is closer). If "
        "unanswerable questions are clearly farther away, a distance cut-off "
        "could refuse them before the model is asked."
    )
    distance_rows = [
        {
            "Questions": label,
            "Closest": stats["min"],
            "Median": stats["median"],
            "Farthest": stats["max"],
        }
        for label, stats in (
            ("Answerable", distances["answerable"]),
            ("Unanswerable", distances["unanswerable"]),
        )
        if stats
    ]
    st.dataframe(
        pd.DataFrame(distance_rows),
        hide_index=True,
        width="stretch",
        column_config={
            name: st.column_config.NumberColumn(format="%.3f")
            for name in ("Closest", "Median", "Farthest")
        },
    )

    st.subheader("Compare the answers")
    with st.expander("Scoring guide"):
        st.markdown(
            "Score an answer by comparing its claims with the cited text and "
            "the expected answer:\n\n"
            "- **5**: every claim is supported by the cited text and matches "
            "the expected answer\n"
            "- **4**: correct, with one small unsupported detail\n"
            "- **3**: partly supported or incomplete\n"
            "- **2**: mostly unsupported or misleading\n"
            "- **1**: wrong, or contradicted by the cited text\n\n"
            "Scores are saved next to the results file and update the summary."
        )

    comparison = question_comparison(report)
    selected = st.selectbox(
        "Question",
        options=range(len(comparison)),
        format_func=lambda i: f"{comparison[i]['id']} | {comparison[i]['question']}",
    )
    item = comparison[selected]
    if item["answerable"]:
        st.info(f"Expected answer: {item['expected_answer'] or 'not recorded'}")
    else:
        st.info(
            "Unanswerable: none of the papers contain the answer, so the "
            "correct behaviour is to refuse."
        )

    columns = st.columns(len(models))
    for column, model in zip(columns, models, strict=True):
        with column:
            render_answer_card(run_path, model, item["answers"][model])


st.title("Evaluation")
st.caption(
    f"Reports are produced by the "
    "evaluation scripts, not by this page."
)

retrieval_tab, llm_tab = st.tabs(["Retrieval", "LLM comparison"])
with retrieval_tab:
    render_retrieval()
with llm_tab:
    render_llm()
