"""Tests for SQLAlchemy database wiring."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app import database


def test_database_engine_supports_sqlite() -> None:
    engine = database.create_database_engine("sqlite:///:memory:")

    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar_one()
    finally:
        engine.dispose()

    assert result == 1


def test_session_factory_is_bound_to_configured_engine() -> None:
    session = database.SessionLocal()

    try:
        assert session.get_bind() is database.engine
    finally:
        session.close()


def test_engine_uses_configured_database_url() -> None:
    assert database.engine.url == make_url(database.settings.database_url)


def test_base_is_a_sqlalchemy_declarative_base() -> None:
    assert issubclass(database.Base, DeclarativeBase)


def test_database_dependency_yields_a_working_session(monkeypatch) -> None:
    engine = database.create_database_engine("sqlite:///:memory:")
    test_session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", test_session_factory)
    dependency = database.get_db()

    try:
        session = next(dependency)
        result = session.execute(text("SELECT 1")).scalar_one()
    finally:
        dependency.close()
        engine.dispose()

    assert result == 1


def test_database_dependency_closes_its_session(monkeypatch) -> None:
    session = Mock(spec=Session)
    monkeypatch.setattr(database, "SessionLocal", lambda: session)
    dependency = database.get_db()

    assert next(dependency) is session
    with pytest.raises(StopIteration):
        next(dependency)

    session.close.assert_called_once_with()


def test_importing_database_module_does_not_create_tables(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "import-check.db"
    environment = os.environ.copy()
    environment["DATABASE_URL"] = f"sqlite:///{database_path.as_posix()}"
    script = (
        "from sqlalchemy import inspect\n"
        "from app.database import engine\n"
        "table_names = inspect(engine).get_table_names()\n"
        "assert table_names == [], "
        "f'Expected no tables after import, found: {table_names}'\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Database import subprocess failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_non_sqlite_engine_omits_sqlite_thread_arguments() -> None:
    database_url = "postgresql://user:password@localhost/incidentiq"
    expected_engine = Mock()

    with patch.object(
        database,
        "create_engine",
        return_value=expected_engine,
    ) as create_engine:
        result = database.create_database_engine(database_url)

    assert result is expected_engine
    create_engine.assert_called_once_with(database_url, connect_args={})
    assert "check_same_thread" not in create_engine.call_args.kwargs["connect_args"]
