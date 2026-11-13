# Development guide

## Setup

Get the code first:

```powershell
git clone https://github.com/teodora-taleska/research_radar.git
cd research_radar
```

Then create the environment (a plain Python 3.11 virtual environment works too):

```powershell
conda create -n research_radar python=3.11
conda activate research_radar
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com) and download the models:

```powershell
ollama pull embeddinggemma
ollama pull qwen3:4b-instruct
```

## Run

Start the backend and the frontend in two terminals from the project root:

```powershell
uvicorn app.api.main:app --reload
```

```powershell
streamlit run frontend/Home.py
```

API documentation: http://127.0.0.1:8000/docs
Application: http://localhost:8501

## Run with Docker

Docker runs the API and the frontend in containers. Ollama is not in a
container: it keeps running on your computer, where the models are already
downloaded and the GPU is available. The containers reach it through
`host.docker.internal`.

Docker only needs three things, and no Python environment: Docker Desktop
running, Ollama running on the host, and the two models downloaded once
(`ollama pull embeddinggemma`, `ollama pull qwen3:4b-instruct`). A copy of the
app started without Docker must not be using ports 8000 and 8501.

1. Install and start Docker Desktop, and start Ollama with the two models
   downloaded (see Setup).
2. From the project root:

```powershell
docker compose up --build
```

3. Open http://localhost:8501 (application) or http://localhost:8000/docs (API).

Uploaded documents and the database are kept in the Docker volume
`research_radar_data`, so they survive `docker compose down` and restarts.
`docker compose down -v` deletes them. The `evaluation/` folder is mounted
into the containers, so benchmark papers and results are shared with the host.

Docker builds the files of the folder you run it in, whatever branch is
checked out there, and the image contains a copy of the code from build time.
After changing code, run `docker compose up --build` again.

If you already run the application without Docker (ports 8000 and 8501 in
use), start the containers on other ports:

```powershell
$env:API_PORT = "18000"; $env:FRONTEND_PORT = "18501"; docker compose up --build
```

Other settings can be overridden the same way, for example
`$env:OLLAMA_LLM_MODEL = "qwen3:4b"`.

On Linux, Ollama must listen on all interfaces so the containers can reach
it: set `OLLAMA_HOST=0.0.0.0` for the Ollama service on the host.

Run the evaluation scripts inside the container (benchmark papers must be in
`evaluation/datasets/papers/`):

```powershell
docker compose run --rm api python -m evaluation.run_evaluation
```

## Configuration

All settings are environment variables with defaults.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | Ollama default | Ollama server address |
| `OLLAMA_EMBEDDING_MODEL` | `embeddinggemma` | Embedding model |
| `OLLAMA_LLM_MODEL` | `qwen3:4b-instruct` | Generative model |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Vector database folder |
| `DATABASE_URL` | `sqlite:///./data/research_radar.db` | Metadata database |
| `RESEARCH_RADAR_API_URL` | `http://127.0.0.1:8000` | Backend address used by the frontend |
| `RESEARCH_RADAR_RESULTS_DIR` | `./evaluation/results` | Folder the Evaluation page reads reports from |

Changing the embedding model requires re-uploading all documents, because
vectors from different models are not comparable.

## Tests

```powershell
python -m pytest tests -q
```

The tests use fake embedders and language models, so they run without
Ollama. Real-model behaviour is checked through the evaluation scripts.

## Lint

```powershell
python -m ruff check app frontend evaluation tests
```
