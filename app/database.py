"""SQLAlchemy engine, session, and declarative base configuration."""

from collections.abc import Generator
from sqlite3 import Connection as SQLiteConnection

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


def _enable_sqlite_foreign_keys(
    dbapi_connection: SQLiteConnection,
    _connection_record: object,
) -> None:
    """Enable SQLite foreign-key constraints for one DBAPI connection."""

    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def create_database_engine(database_url: str) -> Engine:
    """Create an engine configured for the supplied database URL."""

    is_sqlite = make_url(database_url).get_backend_name() == "sqlite"
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    database_engine = create_engine(database_url, connect_args=connect_args)
    if is_sqlite:
        event.listen(database_engine, "connect", _enable_sqlite_foreign_keys)
    return database_engine


class Base(DeclarativeBase):
    """Base class for IncidentIQ persistence models."""


settings = get_settings()
engine = create_database_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """Yield one database session and close it after the request finishes."""

    database_session = SessionLocal()
    try:
        yield database_session
    finally:
        database_session.close()
