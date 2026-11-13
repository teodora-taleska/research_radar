"""Shared services and dependencies for API endpoints."""

from functools import lru_cache

from app.embeddings.provider import OllamaEmbeddingProvider
from app.services.document_service import DocumentService
from app.vectorstore.chroma_store import ChromaDocumentStore


@lru_cache
def get_document_service() -> DocumentService:
    """Return the application-wide document indexing service."""
    return DocumentService(
        embedding_provider=OllamaEmbeddingProvider(),
        store=ChromaDocumentStore(),
    )
