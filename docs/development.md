# Development guide

## Setup

```powershell
conda create -n research_radar python=3.11
conda activate research_radar
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com) and download the models:

```powershell
ollama pull embeddinggemma
ollama pull qwen3:4b
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

## Configuration

All settings are environment variables with defaults.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | Ollama default | Ollama server address |
| `OLLAMA_EMBEDDING_MODEL` | `embeddinggemma` | Embedding model |
| `OLLAMA_LLM_MODEL` | `qwen3:4b` | Generative model |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Vector database folder |
| `DATABASE_URL` | `sqlite:///./data/research_radar.db` | Metadata database |
| `RESEARCH_RADAR_API_URL` | `http://127.0.0.1:8000` | Backend address used by the frontend |

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
