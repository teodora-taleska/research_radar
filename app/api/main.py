# Creates the FastAPI application and registers routes
from fastapi import FastAPI

from app.api.routes import health

app = FastAPI(
    title="ResearchRadar API",
    description="AI-powered research paper assistant",
    version="0.1.0",
)

app.include_router(health.router)


@app.get("/")
def root():
    return {
        "application": "ResearchRadar",
        "message": "AI-powered research paper assistant",
    }