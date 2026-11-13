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

Results on the real benchmark will be recorded here once it is prepared.

A pipeline check on a synthetic benchmark (three 3-page papers, 6 questions)
gave Recall@1 = 1.0 and a median retrieval latency of about 54 ms. This only
confirms that the evaluation works end to end. It says nothing about
quality on real papers.
