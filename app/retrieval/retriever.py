"""Semantic retrieval of relevant document chunks."""

from app.embeddings.provider import EmbeddingProvider
from app.vectorstore.chroma_store import ChromaDocumentStore, StoredChunk


class Retriever:
    """Find the chunks most similar to a question.

    The question is embedded with the same provider used to index the
    documents, so query and chunk vectors live in the same space.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        store: ChromaDocumentStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._store = store

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[StoredChunk]:
        """Return the top-k chunks for a question, most similar first.

        Raises:
            ValueError: If the question is empty or top_k is below 1.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self._embedding_provider.embed_text(question)

        return self._store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_ids=document_ids,
        )
