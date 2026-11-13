"""API endpoint for evidence-grounded research questions."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_research_service
from app.embeddings.provider import EmbeddingProviderError
from app.generation.llm_provider import LLMProviderError
from app.schemas.research import (
    AskRequest,
    AskResponse,
    CitationResponse,
    RetrievedPassage,
)
from app.services.research_service import ResearchService
from app.vectorstore.chroma_store import ChromaStoreError

router = APIRouter(
    prefix="/research",
    tags=["Research"],
)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about the uploaded papers",
    description=(
        "Retrieves the most relevant passages and generates an answer "
        "grounded in them, with document and page citations. If the "
        "documents do not contain enough evidence, says so explicitly."
    ),
)
def ask_question(
    request: AskRequest,
    service: ResearchService = Depends(get_research_service),
) -> AskResponse:
    """Answer a research question using only the indexed documents."""
    try:
        result = service.ask(
            question=request.question,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )
    except (EmbeddingProviderError, LLMProviderError, ChromaStoreError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The question could not be answered: {exc}",
        ) from exc

    return AskResponse(
        question=result.question,
        answer=result.answer.answer,
        sufficient_evidence=result.answer.sufficient_evidence,
        citations=[
            CitationResponse(**vars(citation))
            for citation in result.answer.citations
        ],
        retrieved_passages=[
            RetrievedPassage(
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                distance=chunk.distance,
            )
            for chunk in result.retrieved
        ],
    )
