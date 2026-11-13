"""Database engine and session factory."""

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import Base

DEFAULT_DATABASE_URL = "sqlite:///./data/research_radar.db"


def create_db_engine(database_url: str | None = None) -> Engine:
    """Create an engine and make sure all tables exist.

    The URL comes from the argument, then DATABASE_URL, then the default
    local SQLite file. Switching to another database such as PostgreSQL
    only requires a different URL.
    """
    url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        file_path = url.removeprefix("sqlite:///")
        if file_path and file_path != ":memory:":
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url, connect_args=connect_args)
    Base.metadata.create_all(engine)

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a factory that creates database sessions for an engine."""
    return sessionmaker(bind=engine, expire_on_commit=False)
