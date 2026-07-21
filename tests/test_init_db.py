"""Tests for idempotent database schema initialization."""

import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from app import models
from app.database import create_database_engine


def _run_initialization_script(
    repository_root: Path,
    database_url: str,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DATABASE_URL"] = database_url
    return subprocess.run(
        [sys.executable, "scripts/init_db.py"],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_script_succeeded(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        "Database initialization subprocess failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_init_db_creates_all_tables_and_preserves_existing_data(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "initialized.db"
    database_url = f"sqlite:///{database_path.as_posix()}"

    first_result = _run_initialization_script(repository_root, database_url)
    _assert_script_succeeded(first_result)

    engine = create_database_engine(database_url)
    try:
        assert set(inspect(engine).get_table_names()) == set(
            models.Incident.metadata.tables
        )

        with Session(engine) as session:
            session.add(
                models.Incident(
                    public_id="INC-000001",
                    name="Checkout failures",
                    description="Intermittent checkout errors",
                    affected_service="checkout",
                )
            )
            session.commit()

        second_result = _run_initialization_script(repository_root, database_url)
        _assert_script_succeeded(second_result)

        with Session(engine) as session:
            incident_count = session.scalar(
                select(func.count()).select_from(models.Incident)
            )
    finally:
        engine.dispose()

    assert incident_count == 1
