"""Create all registered IncidentIQ database tables idempotently."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


def initialize_database(database_engine: Engine) -> None:
    """Create missing tables without dropping tables or changing existing rows."""
    from app import models

    models.Incident.metadata.create_all(database_engine)


def main() -> None:
    """Initialize the database configured through IncidentIQ settings."""
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

    from app.database import engine

    initialize_database(engine)


if __name__ == "__main__":
    main()
