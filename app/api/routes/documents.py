# PDF upload, listing and deletion endpoints
"""API endpoints for document upload and ingestion."""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.ingestion.pipeline import process_pdf
from app.schemas.document import (
    ChunkPreview,
    DocumentUploadResponse,
)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process a scientific PDF",
    description=(
        "Extracts PDF text, cleans it, and divides it into "
        "page-aware text chunks."
    ),
)
async def upload_document(
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """Receive a PDF and run the document ingestion pipeline."""
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    try:
        pdf_bytes = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)

        if not pdf_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file is empty.",
            )

        if len(pdf_bytes) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="The PDF exceeds the 20 MB upload limit.",
            )

        try:
            document = process_pdf(
                pdf_bytes=pdf_bytes,
                filename=file.filename,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        chunk_previews = [
            ChunkPreview(
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                character_count=len(chunk.content),
                preview=chunk.content[:240],
            )
            for chunk in document.chunks
        ]

        return DocumentUploadResponse(
            document_id=document.document_id,
            filename=document.filename,
            page_count=document.page_count,
            chunk_count=len(document.chunks),
            chunks=chunk_previews,
        )

    finally:
        await file.close()
