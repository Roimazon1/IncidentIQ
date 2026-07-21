"""Shared Jinja template configuration for application routes."""

from pathlib import Path

from fastapi.templating import Jinja2Templates

from app.models.evidence import SOURCE_NAME_MAX_LENGTH
from app.models.incident import (
    AFFECTED_SERVICE_MAX_LENGTH,
    INCIDENT_NAME_MAX_LENGTH,
)

APP_DIRECTORY = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=APP_DIRECTORY / "templates")
templates.env.globals.update(
    AFFECTED_SERVICE_MAX_LENGTH=AFFECTED_SERVICE_MAX_LENGTH,
    INCIDENT_NAME_MAX_LENGTH=INCIDENT_NAME_MAX_LENGTH,
    SOURCE_NAME_MAX_LENGTH=SOURCE_NAME_MAX_LENGTH,
)
