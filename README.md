# Research Radar

**Scan research. Discover evidence. Save time.**

Research Radar lets you upload scientific PDFs, ask questions in plain
language, and get answers grounded in your papers, with the document and page
for every claim. When the papers do not contain the answer, it says so instead
of guessing.

Everything runs locally: your documents, the embeddings and the language
model stay on your computer (Ollama, ChromaDB, SQLite). No API keys, no cloud.

## Demo

Demo video: *link to be added*. The script is in
[docs/demo_script.md](docs/demo_script.md).

## How it works

![Data flows](docs/images/data-flows.svg)

1. **Indexing:** a PDF is split into page-aware passages, each passage becomes
   a vector (EmbeddingGemma), and passages, vectors and page numbers are
   stored in ChromaDB.
2. **Asking:** the question is embedded with the same model, the closest
   passages are retrieved, and a local language model (Qwen3) writes an
   answer from those passages only, citing them as `[n]`.
3. **Checking:** citations are verified in code against the retrieved
   passages. Invented references are discarded, and an answer with no valid
   citation is reported as insufficient evidence.
4. **Evaluating:** a labelled benchmark measures retrieval (Precision@K,
   Recall@K, latency) and compares language models.

The component view is in [docs/images/architecture.svg](docs/images/architecture.svg)
and explained in [docs/architecture.md](docs/architecture.md).

## Features

- Upload several PDFs and choose which papers to search.
- Answers with citations: file, page and the original passage.
- Explicit "not enough evidence" replies, with all retrieved passages and
  their distances available for inspection.
- Persistent storage: documents survive restarts; papers can be deleted.
- Evaluation page: retrieval metrics, failure cases, run comparison, and a
  side-by-side comparison of language models with manual faithfulness scoring.
- REST API with automatic documentation, 222 automated tests, Docker Compose.

## Results

Measured on a benchmark of 3 papers (two on SPECT cardiac imaging, one on
EEG-fMRI) and 26 questions with relevant pages labelled (22 answerable, 4 not
answerable). Details and caveats: [docs/experiments.md](docs/experiments.md).

**Retrieval** (EmbeddingGemma, top-K passages):

| K | Precision@K | Recall@K |
|---:|---:|---:|
| 1 | 0.64 | 0.34 |
| 3 | 0.62 | 0.77 |
| 5 | 0.54 | 0.89 |
| 10 | 0.41 | 0.96 |

Median retrieval time: 32 ms.

**Language models** (same retrieved passages for both):

| | qwen3:4b | qwen3:4b-instruct |
|---|---:|---:|
| Answered (answerable) | 21 of 22 | 21 of 22 |
| Correct refusals | 4 of 4 | 4 of 4 |
| Median answer time | 22.2 s | 2.4 s |

The instruct model is the default: about 9 times faster, with the same
automatic scores. The automatic checks cannot tell whether a cited claim is
true: the slower model once gave a wrong answer while citing a correct page,
which is why faithfulness is also scored by hand. The benchmark is small, so
treat the numbers as indicative.

## Quick start

Requires Python 3.11 and [Ollama](https://ollama.com).

```powershell
conda create -n research_radar python=3.11
conda activate research_radar
pip install -r requirements.txt
ollama pull embeddinggemma
ollama pull qwen3:4b-instruct
```

Start the backend and the frontend in two terminals:

```powershell
uvicorn app.api.main:app --reload
```

```powershell
streamlit run frontend/Home.py
```

Open http://localhost:8501, choose **Research Assistant**, upload PDFs, click
**Index papers**, and ask a question. API documentation is at
http://127.0.0.1:8000/docs. All settings are environment variables, listed in
[docs/development.md](docs/development.md).

### With Docker

Docker runs the API and the frontend; Ollama keeps running on your computer
with the two models downloaded. With Docker Desktop running:

```powershell
docker compose up --build
```

Then open http://localhost:8501. Uploaded documents are kept in a Docker
volume. Docker Desktop needs a few GB of free memory. See
[docs/development.md](docs/development.md) for ports, settings and running the
evaluation in a container.

## Evaluation

```powershell
python -m evaluation.run_evaluation
python -m evaluation.experiments.llm_comparison
```

The first measures retrieval; the second compares the language models. Results
are saved in `evaluation/results/` and shown on the **Evaluation** page. How to
prepare a benchmark: [evaluation/datasets/README.md](evaluation/datasets/README.md).

## Project structure

```
app/            FastAPI backend
  api/            endpoints and dependency wiring
  ingestion/      PDF extraction, cleaning, chunking
  embeddings/     embedding provider (Ollama)
  vectorstore/    ChromaDB store
  retrieval/      semantic retrieval
  generation/     prompts, LLM provider, citation-checked answers
  services/       document indexing, question answering
  database/       SQLite document records
  schemas/        API request and response models
frontend/       Streamlit app (pages, components, API client)
evaluation/     benchmark, metrics, experiments, results
tests/          unit, integration and frontend tests
docs/           architecture, API, methodology, experiments, demo script
Dockerfile, docker-compose.yml
```

## Tests and code quality

```powershell
python -m pytest tests -q
python -m ruff check .
```

The tests use fake models and run without Ollama.

## Known limitations

- Scanned PDFs are not supported (no OCR).
- Each question is answered independently, and the chat is lost when the
  browser tab is refreshed.
- Whether the model refuses unanswerable questions depends on the model;
  retrieval distance alone does not separate them.
- Citations are verified to exist, not to be faithful to the claim.
- The benchmark is small (3 papers, 26 questions), written by the author, and
  relevance is judged per page, which is lenient.
- Docker packages only the API and the frontend; Ollama and its models must be
  installed separately.
- Local models on a laptop are slow with thinking models and need free memory.

## Roadmap (version 2)

- Paper Explorer (structured summaries) and Paper Comparison pages.
- Saved conversations and follow-up questions.
- PostgreSQL with migrations; hybrid search (BM25) and reranking.
- Public hosted demo with a hosted model provider; optional containerised Ollama.
- OCR for scanned PDFs; larger benchmark with more metrics (MRR, nDCG);
  chunk-size and embedding-model comparisons.

## Documentation

- [docs/architecture.md](docs/architecture.md): components and data flows
- [docs/api.md](docs/api.md): REST endpoints
- [docs/methodology.md](docs/methodology.md): methods and evaluation
- [docs/experiments.md](docs/experiments.md): experiments and results
- [docs/development.md](docs/development.md): setup, configuration, Docker
- [docs/demo_script.md](docs/demo_script.md): demo video script

## License

MIT, see [LICENSE](LICENSE).
