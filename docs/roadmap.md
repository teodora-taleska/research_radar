# Roadmap: version 2

Features that are **not implemented** in version 1. For each one: what it is,
where it would live, what it builds on, and how to check that it works. File
names are planned, not existing, unless a page or module is named as existing.

Version 1 (done) covers PDF upload, indexing, cited answers, refusal when the
papers lack evidence, document management, the Research Assistant page, the
Evaluation page, and Docker. See the [README](../README.md) for what exists.

## Product features

### Saved conversations and follow-up questions
- **What:** keep the questions, answers and citations of a conversation, so
  they survive a browser refresh, and understand follow-ups such as "and what
  were its limitations?".
- **Where:** `app/api/routes/chat.py`, `app/schemas/chat.py`,
  `app/services/conversation_service.py`, new `conversations` and `messages`
  tables in `app/database/models.py`, an optional `conversation_id` on
  `POST /research/ask`, and a conversation list in the Research Assistant
  sidebar.
- **Builds on:** `ResearchService` and the SQLite repository pattern in
  `app/database/repositories.py`.
- **Care needed:** follow-ups need the question rewritten using earlier turns
  before retrieval. That can hurt citation correctness, so evaluate it on a
  multi-turn question set.

### Paper Explorer (page 2)
- **What:** pick a paper and get a structured summary: objective, methodology,
  datasets, models, metrics, results, limitations. Every field must cite the
  passages it came from.
- **Where:** `frontend/pages/2_Paper_Explorer.py` (a placeholder page exists),
  `frontend/components/paper_cards.py`, a new service for paper analysis, and
  prompts in `app/generation/prompts.py`.
- **Builds on:** retrieval restricted to one document (`document_ids`) and the
  citation checking in `AnswerGenerator`.
- **Check:** compare the extracted fields with hand-written ones for a few
  papers.

### Paper Comparison (page 3)
- **What:** compare several papers field by field and point out similarities,
  contradictions and research gaps, each backed by cited evidence.
- **Where:** `frontend/pages/3_Paper_Comparison.py` (placeholder page exists),
  reusing the paper analysis service.
- **Care needed:** a comparison is only as reliable as its evidence; show the
  cited passages next to every claim.

## Retrieval and answer quality

### Hybrid search and reranking
- **What:** combine semantic search with keyword search (BM25), and rerank the
  retrieved passages (for example with a cross-encoder) before generation.
- **Where:** `app/retrieval/hybrid_search.py`, `app/retrieval/reranker.py`,
  used by `app/retrieval/retriever.py`.
- **Check:** `python -m evaluation.run_evaluation`; the Evaluation page already
  compares runs. Weakest questions today: q17, q18, q19, q20 (facts spread
  across pages).

### Better refusal of unanswerable questions
- **Finding:** retrieval distance does not separate answerable from
  unanswerable questions (see [experiments](experiments.md)).
- **Options:** a reranker score threshold, or asking the model first whether
  the passages contain the answer.
- **Check:** a larger set of unanswerable questions, including ones close to the
  papers' topics.

### Retrieval experiments
- **What:** compare chunk sizes and overlaps, and other embedding models.
- **How:** `run_evaluation` already accepts `--chunk-size` and `--chunk-overlap`,
  and the embedding model is `OLLAMA_EMBEDDING_MODEL`. Run each variant and
  compare on the Evaluation page (run comparison table).

### More and better evaluation
- Add more papers and questions, labelled by someone other than the author.
- Add MRR and nDCG to `evaluation/metrics.py`.
- Score faithfulness with more than one person.

## Engineering

| Item | Detail |
|---|---|
| PostgreSQL and migrations | Set `DATABASE_URL`; add Alembic migrations (Alembic is already in `requirements.txt`, tables are created automatically today) |
| Central settings and logging | A settings module (for example with pydantic-settings) instead of environment variables read in each module; structured logging |
| Continuous integration | A GitHub Actions workflow running `ruff check .` and `pytest` |
| OCR | Support for scanned PDFs |
| Original PDFs | Keep the uploaded file so a citation can open the PDF at its page |
| Helper scripts | Download the benchmark papers listed in `evaluation/datasets/papers.md`; reset the local data for a clean start |

## Deployment

- **Public hosted demo:** a hosted model provider implementing the existing
  `EmbeddingProvider` and `LLMProvider` interfaces (for example Groq), deployed
  with the Docker setup on a service that offers persistent storage. Vercel does
  not fit because it has no persistent disk and cannot run Streamlit.
- **Docker without a separate Ollama install:** an optional compose profile
  with an Ollama container and a one-time model download. It runs on the CPU
  unless GPU passthrough is configured, needs a download of about 3 GB and
  more memory.

## Suggested order

1. Larger benchmark and a better refusal mechanism (they shape every later
   decision).
2. Saved conversations.
3. Paper Explorer, then Paper Comparison.
4. Hybrid search and reranking, measured with the evaluation.
5. PostgreSQL, CI, hosted demo.
