# Research Radar

Scan research. Discover evidence. Save time.

## Overview

Reading scientific literature is slow. Research Radar lets you upload
scientific PDFs, ask questions in plain language, and get answers grounded in
your papers, with the document and page for every claim. When the papers do
not contain the answer, it says so instead of guessing.

Everything runs locally: documents, embeddings and the language model stay on
your computer (Ollama, ChromaDB, SQLite), and no API keys are needed.

## Status

This is the first working version (MVP).

| Feature | Status |
|---|---|
| PDF upload, text extraction, cleaning, page-aware chunking | Done |
| Local embeddings (EmbeddingGemma) and persistent ChromaDB index | Done |
| Semantic retrieval with cosine distances | Done |
| Local LLM answers (Qwen3) with verified page citations | Done |
| Explicit "insufficient evidence" answers | Done |
| Document records, listing and deletion (SQLite) | Done |
| Research Assistant page (Streamlit) | Done |
| Retrieval evaluation engine (Precision@K, Recall@K, latency) | Done |
| Evaluation page with retrieval metrics and LLM comparison | Done |
| Docker, hosted demo with Groq, architecture diagram | Planned |
| Paper Explorer and Paper Comparison pages | Version 2 |
| Saved conversations, follow-up questions, PostgreSQL, hybrid search, reranking | Version 2 |

## How it works

```
PDF -> extract and clean -> page-aware chunks -> embeddings -> ChromaDB
Question -> embedding -> top-K passages -> LLM with numbered sources
         -> answer + verified citations (document, page, original text)
```

Citations are checked in code against the retrieved passages: invented
references are discarded, and an answer without a valid citation is reported
as insufficient evidence. See [docs/architecture.md](docs/architecture.md).

## Technology stack

Python 3.11, FastAPI, Streamlit, Ollama (EmbeddingGemma, Qwen3), ChromaDB,
SQLAlchemy with SQLite, PyMuPDF, LangChain text splitters, pytest, Ruff.

## Installation

Requires Python 3.11 and [Ollama](https://ollama.com).

```powershell
conda create -n research_radar python=3.11
conda activate research_radar
pip install -r requirements.txt
ollama pull embeddinggemma
ollama pull qwen3:4b-instruct
```

## Running

Start the backend and the frontend in two terminals:

```powershell
uvicorn app.api.main:app --reload
```

```powershell
streamlit run frontend/Home.py
```

Open http://localhost:8501, choose **Research Assistant**, upload PDFs, click
**Index papers**, and ask a question. API documentation is at
http://127.0.0.1:8000/docs. Configuration options are listed in
[docs/development.md](docs/development.md).

## Running tests

```powershell
python -m pytest tests -q
```

The tests use fake models and run without Ollama.

## Evaluation

Retrieval quality is measured on a labelled benchmark of papers and questions
(Precision@K, Recall@K, latency):

```powershell
python -m evaluation.run_evaluation
```

Two language models can be compared on the same questions (speed, citations,
refusing unanswerable questions, and a manual faithfulness score):

```powershell
python -m evaluation.experiments.llm_comparison
```

Results are shown on the **Evaluation** page of the app. How to prepare the
benchmark: [evaluation/datasets/README.md](evaluation/datasets/README.md).
Method and results: [docs/methodology.md](docs/methodology.md) and
[docs/experiments.md](docs/experiments.md).

## Known limitations

- Scanned PDFs are not supported (no OCR).
- Each question is answered independently, and the chat is lost when the
  browser tab is refreshed.
- Whether the model refuses unanswerable questions depends on the model.
- Citations are verified to exist, not to be faithful to the claim.
- Only tested on small examples so far; results on real papers are still to
  be measured.

## Documentation

- [docs/architecture.md](docs/architecture.md): components and data flows
- [docs/api.md](docs/api.md): REST endpoints
- [docs/methodology.md](docs/methodology.md): methods and evaluation
- [docs/experiments.md](docs/experiments.md): how to run experiments
- [docs/development.md](docs/development.md): setup and configuration
