"""Unit tests for the SQLite document repository and its persistence."""

from app.database.connection import create_db_engine, create_session_factory
from app.database.repositories import DocumentRepository


def test_add_and_get(repository):
    repository.add("doc1", "paper.pdf", page_count=4, chunk_count=12)

    record = repository.get("doc1")

    assert record.filename == "paper.pdf"
    assert record.page_count == 4
    assert record.chunk_count == 12
    assert record.status == "indexed"


def test_get_missing_returns_none(repository):
    assert repository.get("ghost") is None


def test_delete(repository):
    repository.add("doc1", "paper.pdf", 1, 1)

    assert repository.delete("doc1") is True
    assert repository.get("doc1") is None
    assert repository.delete("doc1") is False


def test_data_survives_reopening_the_database(tmp_path):
    url = f"sqlite:///{(tmp_path / 'persist.db').as_posix()}"
    first_engine = create_db_engine(url)
    DocumentRepository(create_session_factory(first_engine)).add(
        "doc1", "paper.pdf", 2, 5
    )
    first_engine.dispose()

    second_engine = create_db_engine(url)
    reopened = DocumentRepository(create_session_factory(second_engine))

    assert [d.document_id for d in reopened.list_all()] == ["doc1"]
    second_engine.dispose()


def test_creates_missing_database_directory(tmp_path):
    url = f"sqlite:///{(tmp_path / 'nested' / 'dir' / 'app.db').as_posix()}"

    engine = create_db_engine(url)
    engine.dispose()

    assert (tmp_path / "nested" / "dir" / "app.db").exists()
