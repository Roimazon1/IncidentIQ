"""SQLAlchemy engine, session, and declarative base configuration."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


def create_database_engine(database_url: str) -> Engine:
    """Create an engine configured for the supplied database URL."""

    connect_args = (
        {"check_same_thread": False}
        if make_url(database_url).get_backend_name() == "sqlite"
        else {}
    )
    return create_engine(database_url, connect_args=connect_args)


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
