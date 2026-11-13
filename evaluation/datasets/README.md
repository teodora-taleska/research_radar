# Retrieval benchmark

This folder holds the benchmark used to measure retrieval quality.

```
evaluation/datasets/
  papers/           PDF papers (not committed, see below)
  questions.json    labelled questions
```

## 1. Choose papers

Pick 3 to 5 open-access papers (for example from arXiv). Use papers that
overlap in topic so retrieval has to tell similar documents apart. Put the
PDFs in `evaluation/datasets/papers/`. Every PDF in the folder is indexed,
including ones no question refers to.

PDFs are git-ignored to keep the repository small and avoid redistributing
papers. List their sources (title and arXiv ID or URL) in the project
report so others can download the same files.

## 2. Write questions

Write 15 to 20 questions a researcher might ask, such as:

- What dataset was used for evaluation?
- Which metrics were reported?
- What are the limitations of the method?
- How does the method differ from the baseline?

Include 3 to 5 **unanswerable** questions that none of the papers can answer.
They are skipped by the retrieval metrics and used later to check that the
assistant refuses instead of guessing.

## 3. Label relevant pages

For every answerable question, open the papers and note the page or pages
that contain the answer. Use the **PDF viewer's page counter**, starting
at 1, not the page number printed on the paper (front matter shifts it).

Create `questions.json`:

```json
{
  "questions": [
    {
      "id": "q01",
      "question": "What dataset was used for evaluation?",
      "relevant": [
        {"paper": "paper_a.pdf", "pages": [4, 5]}
      ]
    },
    {
      "id": "q02",
      "question": "How do the two methods differ?",
      "relevant": [
        {"paper": "paper_a.pdf", "pages": [3]},
        {"paper": "paper_b.pdf", "pages": [2, 3]}
      ]
    },
    {
      "id": "q03",
      "question": "Who won the 2018 football World Cup?",
      "answerable": false
    }
  ]
}
```

Rules checked automatically when you run the evaluation:

- `id` values are unique.
- Every `paper` exists in `papers/`.
- Page numbers are integers of 1 or more and exist in that paper.
- Answerable questions list at least one page; unanswerable ones list none.

## 4. Run

With Ollama running, from the project root:

```powershell
python -m evaluation.run_evaluation
```

Options: `--ks 1 3 5 10`, `--chunk-size 1200`, `--chunk-overlap 200`,
`--questions PATH`, `--papers PATH`, `--output-dir PATH`.

Results are saved as JSON in `evaluation/results/`.

## How a question is scored

A retrieved passage is **relevant** when its (paper, page) is one of the
labelled pages.

- **Precision@K**: relevant passages among the top K, divided by K.
- **Recall@K**: labelled pages found in the top K, divided by the number
  of labelled pages.
- **Latency**: time to embed the question and search, per question, after
  a warm-up call so model loading is not counted.

Because relevance is judged per page, a passage from the right page counts
even if it is not the exact sentence with the answer. This is lenient, and
it keeps labels valid when chunk size changes.
