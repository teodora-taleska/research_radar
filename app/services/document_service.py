"""Indexing and management of uploaded documents."""

from app.core.exceptions import DocumentNotFoundError
from app.database.models import DocumentRecord
from app.database.repositories import DocumentRepository
from app.embeddings.provider import EmbeddingProvider
from app.ingestion.pipeline import ProcessedDocument
from app.vectorstore.chroma_store import ChromaDocumentStore, ChromaStoreError

EMBEDDING_BATCH_SIZE = 32


class DocumentService:
    """Index documents into the vector store and track their metadata."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        store: ChromaDocumentStore,
        repository: DocumentRepository,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self._embedding_provider = embedding_provider
        self._store = store
        self._repository = repository
        self._batch_size = batch_size

    def index_document(self, document: ProcessedDocument) -> int:
        """Embed all chunks, store them, and record the document.

        Every embedding is generated before anything is written, so an
        embedding failure leaves everything untouched. If storing the
        vectors or the metadata fails, the document is removed again so
        the vector store and the database never disagree.

        Returns:
            Number of chunks indexed.
        """
        chunks = document.chunks
        embeddings: list[list[float]] = []

        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            embeddings.extend(
                self._embedding_provider.embed_texts(
                    [chunk.content for chunk in batch]
                )
            )

        chunk_dicts = [
            {
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "content": chunk.content,
            }
            for chunk in chunks
        ]

        try:
            indexed = self._store.add_chunks(
                document_id=document.document_id,
                filename=document.filename,
                chunks=chunk_dicts,
                embeddings=embeddings,
            )
        except ChromaStoreError:
            self._store.delete_document(document.document_id)
            raise

        try:
            self._repository.add(
                document_id=document.document_id,
                filename=document.filename,
                page_count=document.page_count,
                chunk_count=indexed,
            )
        except Exception:
            self._store.delete_document(document.document_id)
            raise

        return indexed

    def list_documents(self) -> list[DocumentRecord]:
        """Return all uploaded documents, newest first."""
        return self._repository.list_all()

    def get_document(self, document_id: str) -> DocumentRecord:
        """Return one document.

        Raises:
            DocumentNotFoundError: If the document does not exist.
        """
        record = self._repository.get(document_id)
        if record is None:
            raise DocumentNotFoundError(f"Document '{document_id}' not found.")
        return record

    def delete_document(self, document_id: str) -> None:
        """Delete a document's vectors and its metadata.

        Raises:
            DocumentNotFoundError: If the document does not exist.
            ChromaStoreError: If its vectors could not be removed. The
                metadata is kept so the deletion can be retried.
        """
        self.get_document(document_id)
        self._store.delete_document(document_id)
        self._repository.delete(document_id)
