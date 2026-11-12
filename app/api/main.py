# Creates the FastAPI application and registers routes
from fastapi import FastAPI

from app.api.routes import documents, health

app = FastAPI(
    title="Research Radar API",
    description="AI-powered research paper intelligence assistant",
    version="0.2.0",
)

app.include_router(health.router)
app.include_router(documents.router)


@app.get("/")
def root():
    """Return basic application information."""
    return {
        "application": "Research Radar",
        "message": "Research paper intelligence assistant",
    }
