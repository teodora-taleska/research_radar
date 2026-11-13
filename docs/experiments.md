# Experiments

## Retrieval evaluation

Measures Precision@K, Recall@K and latency on a labelled benchmark.

1. Prepare the benchmark: see `evaluation/datasets/README.md`.
2. Run with Ollama running:

```powershell
python -m evaluation.run_evaluation
```

The papers are indexed into a temporary store, so uploaded documents are
not affected. Each run writes `evaluation/results/retrieval_<timestamp>.json`
containing the configuration (embedding model, chunk size and overlap, K
values, papers), summary metrics, and for every question the retrieved
pages with their distances. Questions with no relevant page in the top K
are reported as failure cases.

Parameters can be varied from the command line, for example
`--chunk-size 800 --chunk-overlap 100`.

## Results

### Baseline retrieval (13 November 2026)

Benchmark: 3 papers (38 PDF pages, 203 indexed chunks) and 22 answerable
questions with 45 labelled relevant pages in total (about 2 per question).
Configuration: EmbeddingGemma (768 dimensions), cosine distance, chunk size
1200, overlap 200. Raw results: `evaluation/results/retrieval_20261113_165622.json`.

| K | Precision@K | Recall@K |
|---:|---:|---:|
| 1 | 0.636 | 0.339 |
| 3 | 0.621 | 0.770 |
| 5 | 0.536 | 0.891 |
| 10 | 0.405 | 0.955 |

Retrieval latency (question embedding plus search): mean 36 ms, median 32 ms,
95th percentile 52 ms.

How to read this:

- Recall@1 is low by design. Most questions have more than one relevant
  page, so one result can find at most one of them.
- Recall@5 of 0.89 means that for most questions the top 5 passages contain
  the relevant pages, which is what the language model receives by default.
- The retriever picked the right paper for nearly every question, including
  between the two SPECT papers. Most misses are neighbouring pages of the
  correct paper, or the same fact repeated on another page of a long paper.
- Weakest questions: q18 (assumptions of average artifact subtraction) and
  q20 (best method for alpha and beta bands) reach only Recall@10 = 0.5,
  because the facts are spread across several pages of the 20-page EEG paper.

Caveats: 22 questions and 3 papers is a small sample, so differences of a few
percentage points are not meaningful. Relevance is judged per page, which is
lenient (a page counts even if the answer sentence is elsewhere on it) and
also strict (a page that only partly answers the question does not count).
Questions were written and labelled by the same author who built the system,
which can bias them towards easy phrasing.

Not yet measured here: other chunk sizes or embedding models. Answer quality
is covered by the LLM comparison below.

## LLM comparison

Compares language models on the same benchmark. Every question (22
answerable, 4 unanswerable) is answered with each model, using the same
retrieved top-5 passages, so differences come from the model alone.

Both models must be installed in Ollama (`ollama pull qwen3:4b` and
`ollama pull qwen3:4b-instruct`). Run:

```powershell
python -m evaluation.experiments.llm_comparison
```

Options: `--models A B`, `--top-k 5`, `--chunk-size`, `--chunk-overlap`. Each
model is loaded before its questions, so loading time is not counted. The
report is saved as `evaluation/results/llm_comparison_<timestamp>.json`.

Measured automatically:

| Metric | Meaning |
|---|---|
| Answered (answerable) | Share of answerable questions that received a cited answer |
| Correct refusals | Share of unanswerable questions where the model declined |
| Citation on relevant page | Share of answers with at least one citation to a labelled relevant page |
| Citation precision | Share of all citations that point to a labelled relevant page |
| Latency | Seconds per answer: mean, median and 95th percentile |

"Refused" means the answer was reported as insufficient evidence. That also
covers answers rejected because they cited no retrieved passage.

Measured by hand: **faithfulness**, a score from 1 to 5 of whether every claim
in an answer is supported by the cited text and matches the expected answer
(scoring guide on the Evaluation page). Scores are entered on the Evaluation
page, saved as `evaluation/results/faithfulness_<timestamp>.json`, and
averaged per model in the summary table.

The report also records the distance of the best retrieved passage for
answerable and unanswerable questions. If unanswerable questions are clearly
farther away, a distance cut-off could refuse them before the model is asked.

### Results (13 November 2026)

Raw results: `evaluation/results/llm_comparison_20261113_171829.json`. Same 3
papers, 26 questions (22 answerable, 4 unanswerable), top-5 passages,
temperature 0.1.

| | qwen3:4b | qwen3:4b-instruct |
|---|---:|---:|
| Answered (answerable) | 95% (21/22) | 95% (21/22) |
| Correct refusals | 100% (4/4) | 100% (4/4) |
| Citation on relevant page | 100% | 100% |
| Citation precision | 0.94 | 0.94 |
| Median answer time | 22.2 s | 2.4 s |
| 95th percentile answer time | 81.6 s | 3.7 s |

Findings:

- **Speed:** the instruct model is about 9 times faster at the median and
  more than 20 times faster at the 95th percentile. `qwen3:4b` is a thinking
  model that writes long reasoning before answering. The `qwen3:4b` run
  shared the machine with other work, but the gap is far larger than that
  could explain.
- **Automatic metrics are equal.** Each model declined one answerable
  question: `qwen3:4b` q16 (participant numbers) after 88 s, the instruct
  model q20 (best method for alpha and beta bands).
- **Citation checks cannot detect wrong claims.** On q20 `qwen3:4b` answered
  that OBS was best, citing a labelled relevant page, but the paper reports
  CWL as best; it also gave wrong expansions of the acronyms. This counts as
  a correct, well-cited answer in every automatic metric. Faithfulness has to
  be judged by a person (not yet scored for this run).
- **Retrieval distance does not separate unanswerable questions.** The best
  passage had a median distance of 0.42 for answerable questions (range
  0.28 to 0.66) and 0.47 for unanswerable ones (range 0.33 to 0.81). Only the
  off-topic football question was clearly far (0.81). Unanswerable questions
  about the same topic (q24, q25) retrieved passages as close as answerable
  ones, so a distance cut-off would not replace the model's refusal.

Decision: the default model is now `qwen3:4b-instruct`. It is about 9 times
faster, matches `qwen3:4b` on every automatic metric, and the one clearly
wrong answer in this run came from `qwen3:4b`. This was decided before the
manual faithfulness scores were entered, so it rests on speed, the automatic
metrics and the answers inspected. The scores can confirm or overturn it
(`OLLAMA_LLM_MODEL=qwen3:4b` restores the previous model).

Caveats: 4 unanswerable questions, 22 answerable questions, one run per
model, one machine. Differences of one question are not meaningful.

## Evaluation page

The Evaluation page (`streamlit run frontend/Home.py`, then Evaluation) reads
the reports from `evaluation/results/` (override with
`RESEARCH_RADAR_RESULTS_DIR`). It shows, for the retrieval runs: Precision@K,
Recall@K and latency for a chosen K, a chart by K, results per question,
failure cases with the retrieved pages and distances, and a comparison table
when several runs exist (for example different chunk sizes). For the LLM
comparison it shows the summary table, speed and behaviour charts, the
distance analysis, and the answers of all models side by side with the
expected answer and a faithfulness score selector.
