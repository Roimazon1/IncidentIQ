"""Shared Pytest fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app import models
from app.database import create_database_engine, get_db
from app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide a test client for the IncidentIQ application."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def database_session_factory(
    tmp_path: Path,
) -> Iterator[sessionmaker[Session]]:
    """Provide isolated file-backed sessions with application engine settings."""
    database_path = tmp_path / "incidentiq-test.db"
    engine = create_database_engine(f"sqlite:///{database_path.as_posix()}")
    models.Incident.metadata.create_all(engine)
    test_session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
    try:
        yield test_session_factory
    finally:
        engine.dispose()


@pytest.fixture
def database_client(
    database_session_factory: sessionmaker[Session],
) -> Iterator[TestClient]:
    """Provide an application client bound to the isolated test database."""

    def override_get_db() -> Iterator[Session]:
        with database_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
