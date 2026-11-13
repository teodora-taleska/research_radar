"""Indexing of processed documents into the vector store."""

from app.embeddings.provider import EmbeddingProvider
from app.ingestion.pipeline import ProcessedDocument
from app.vectorstore.chroma_store import ChromaDocumentStore, ChromaStoreError

EMBEDDING_BATCH_SIZE = 32


class DocumentService:
    """Embed document chunks and store them in the vector store."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        store: ChromaDocumentStore,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self._embedding_provider = embedding_provider
        self._store = store
        self._batch_size = batch_size

    def index_document(self, document: ProcessedDocument) -> int:
        """Embed all chunks of a document and store them.

        Every embedding is generated before anything is written, so an
        embedding failure leaves the store untouched. If the write itself
        fails, any partially stored chunks are removed.

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
            return self._store.add_chunks(
                document_id=document.document_id,
                filename=document.filename,
                chunks=chunk_dicts,
                embeddings=embeddings,
            )
        except ChromaStoreError:
            self._store.delete_document(document.document_id)
            raise
