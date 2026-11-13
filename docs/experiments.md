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

Not yet measured: answer quality, citation faithfulness and refusal on the 4
unanswerable questions (planned LLM comparison), and other chunk sizes or
embedding models.
