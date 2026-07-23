"""Settings-driven construction for the report service facade."""

from sqlalchemy.orm import Session

from app.config import Settings
from app.services.analysis_service_factory import build_configured_ai_provider
from app.services.report_service import ReportService


def build_configured_report_service(
    session: Session,
    settings: Settings,
) -> ReportService:
    """Build a report service with the settings-selected provider."""
    return ReportService(
        session,
        ai_provider=build_configured_ai_provider(settings),
    )
