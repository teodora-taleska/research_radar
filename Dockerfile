FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY app ./app
COPY frontend ./frontend
COPY evaluation ./evaluation

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app/data
USER appuser

ENV CHROMA_PERSIST_DIR=/app/data/chroma \
    DATABASE_URL=sqlite:////app/data/research_radar.db

EXPOSE 8000 8501

# The API by default. The frontend service overrides the command.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
