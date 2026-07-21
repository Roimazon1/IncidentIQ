"""Shared fixtures for isolated persistence model tests."""

from collections.abc import Iterator

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app import models
from app.database import create_database_engine


@pytest.fixture
def sqlite_engine() -> Iterator[Engine]:
    """Provide a fresh in-memory SQLite engine for each test."""

    engine = create_database_engine("sqlite:///:memory:")
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def model_session_factory(sqlite_engine: Engine) -> sessionmaker[Session]:
    """Provide sessions backed by fresh IncidentIQ model tables."""

    models.Incident.metadata.create_all(sqlite_engine)
    return sessionmaker(bind=sqlite_engine)
