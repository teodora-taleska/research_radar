"""Shared services and dependencies for API endpoints."""

from functools import lru_cache

from app.embeddings.provider import OllamaEmbeddingProvider
from app.generation.answer_generator import AnswerGenerator
from app.generation.llm_provider import OllamaLLMProvider
from app.retrieval.retriever import Retriever
from app.services.document_service import DocumentService
from app.services.research_service import ResearchService
from app.vectorstore.chroma_store import ChromaDocumentStore


@lru_cache
def get_embedding_provider() -> OllamaEmbeddingProvider:
    """Return the shared embedding provider."""
    return OllamaEmbeddingProvider()


@lru_cache
def get_vector_store() -> ChromaDocumentStore:
    """Return the shared persistent vector store."""
    return ChromaDocumentStore()


@lru_cache
def get_document_service() -> DocumentService:
    """Return the application-wide document indexing service."""
    return DocumentService(
        embedding_provider=get_embedding_provider(),
        store=get_vector_store(),
    )


@lru_cache
def get_research_service() -> ResearchService:
    """Return the application-wide research question service."""
    return ResearchService(
        retriever=Retriever(get_embedding_provider(), get_vector_store()),
        answer_generator=AnswerGenerator(OllamaLLMProvider()),
    )
