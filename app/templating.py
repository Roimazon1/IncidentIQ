"""Shared Jinja template configuration for application routes."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.models.evidence import SOURCE_NAME_MAX_LENGTH
from app.models.incident import (
    AFFECTED_SERVICE_MAX_LENGTH,
    INCIDENT_NAME_MAX_LENGTH,
)
from app.success_notices import (
    SUCCESS_NOTICE_QUERY_PARAMETER,
    success_notice_message,
)

APP_DIRECTORY = Path(__file__).resolve().parent


def format_incident_datetime(value: datetime) -> str:
    """Format an aware incident timestamp in the configured display timezone."""
    display_timezone = ZoneInfo(get_settings().display_timezone)
    return value.astimezone(display_timezone).strftime("%b %d, %Y at %H:%M %Z")


templates = Jinja2Templates(directory=APP_DIRECTORY / "templates")
templates.env.filters["incident_datetime"] = format_incident_datetime
templates.env.globals.update(
    AFFECTED_SERVICE_MAX_LENGTH=AFFECTED_SERVICE_MAX_LENGTH,
    INCIDENT_NAME_MAX_LENGTH=INCIDENT_NAME_MAX_LENGTH,
    SOURCE_NAME_MAX_LENGTH=SOURCE_NAME_MAX_LENGTH,
    SUCCESS_NOTICE_QUERY_PARAMETER=SUCCESS_NOTICE_QUERY_PARAMETER,
    success_notice_message=success_notice_message,
)
