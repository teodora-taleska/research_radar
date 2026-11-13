"""Persistent ChromaDB vector store for document chunks."""

import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings


class ChromaStoreError(RuntimeError):
    """Raised when a ChromaDB operation fails."""


@dataclass(frozen=True)
class StoredChunk:
    """A document chunk retrieved from the vector store."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    content: str
    distance: float


class ChromaDocumentStore:
    """Persistent vector store for scientific document chunks.

    Stores embeddings, text and metadata for all uploaded documents.
    Uses cosine distance for similarity search.
    """

    COLLECTION_NAME = "research_radar_documents"

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            directory = persist_directory or os.getenv(
                "CHROMA_PERSIST_DIR",
                "./data/chroma",
            )
            Path(directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(directory),
                settings=Settings(anonymized_telemetry=False),
            )

        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """Store chunks and their embeddings for one document.

        Args:
            document_id: Unique identifier for the source document.
            filename: Original PDF filename.
            chunks: List of dicts with keys: chunk_index, page_number, content.
            embeddings: One embedding vector per chunk, same order as chunks.

        Returns:
            Number of chunks stored.

        Raises:
            ChromaStoreError: If chunks and embeddings lengths differ, or on
                any ChromaDB failure.
        """
        if len(chunks) != len(embeddings):
            raise ChromaStoreError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        if not chunks:
            return 0

        ids = [
            f"{document_id}__chunk_{chunk['chunk_index']}"
            for chunk in chunks
        ]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ]

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise ChromaStoreError(
                f"Failed to add chunks for document '{document_id}': {exc}"
            ) from exc

        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks belonging to a document.

        Args:
            document_id: The document whose chunks should be deleted.

        Returns:
            Number of chunks removed.
        """
        try:
            existing = self._collection.get(
                where={"document_id": document_id}
            )
            count = len(existing["ids"])
            if count:
                self._collection.delete(where={"document_id": document_id})
            return count
        except Exception as exc:
            raise ChromaStoreError(
                f"Failed to delete document '{document_id}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[StoredChunk]:
        """Find the most similar chunks to a query embedding.

        Args:
            query_embedding: Embedding of the search query.
            top_k: Maximum number of results to return.
            document_ids: If provided, restrict search to these documents.

        Returns:
            List of StoredChunk objects ordered by ascending distance
            (most similar first).
        """
        where = None
        if document_ids:
            if len(document_ids) == 1:
                where = {"document_id": document_ids[0]}
            else:
                where = {"document_id": {"$in": document_ids}}

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count() or 1),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise ChromaStoreError(f"Search failed: {exc}") from exc

        chunks: list[StoredChunk] = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, content, meta, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            chunks.append(
                StoredChunk(
                    chunk_id=chunk_id,
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    page_number=meta["page_number"],
                    chunk_index=meta["chunk_index"],
                    content=content,
                    distance=distance,
                )
            )

        return chunks

    def list_documents(self) -> list[dict]:
        """Return one record per unique document in the store.

        Returns:
            List of dicts with keys: document_id, filename, chunk_count.
        """
        try:
            all_items = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise ChromaStoreError(
                f"Failed to list documents: {exc}"
            ) from exc

        seen: dict[str, dict] = {}
        for meta in all_items["metadatas"]:
            doc_id = meta["document_id"]
            if doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta["filename"],
                    "chunk_count": 0,
                }
            seen[doc_id]["chunk_count"] += 1

        return list(seen.values())

    def document_exists(self, document_id: str) -> bool:
        """Return True if the store contains any chunks for this document."""
        try:
            result = self._collection.get(
                where={"document_id": document_id},
                limit=1,
            )
            return len(result["ids"]) > 0
        except Exception as exc:
            raise ChromaStoreError(
                f"Failed to check document '{document_id}': {exc}"
            ) from exc

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        return self._collection.count()
