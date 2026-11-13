"""Data access for document metadata."""

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import DocumentRecord


class DocumentRepository:
    """Store and retrieve document metadata."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def add(
        self,
        document_id: str,
        filename: str,
        page_count: int,
        chunk_count: int,
        status: str = "indexed",
    ) -> DocumentRecord:
        """Save a new document record."""
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            page_count=page_count,
            chunk_count=chunk_count,
            status=status,
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()
        return record

    def get(self, document_id: str) -> DocumentRecord | None:
        """Return one document record, or None if it does not exist."""
        with self._session_factory() as session:
            return session.get(DocumentRecord, document_id)

    def list_all(self) -> list[DocumentRecord]:
        """Return all documents, newest first."""
        with self._session_factory() as session:
            statement = select(DocumentRecord).order_by(
                DocumentRecord.created_at.desc()
            )
            return list(session.scalars(statement))

    def delete(self, document_id: str) -> bool:
        """Delete a document record. Returns False if it did not exist."""
        with self._session_factory() as session:
            record = session.get(DocumentRecord, document_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            return True
