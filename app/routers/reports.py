"""Incident-scoped postmortem generation, preview, and draft-save routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.routers.validation import validation_messages
from app.schemas.report import ReportDraftUpdate
from app.services.analysis_service import AnalysisRunNotFoundError
from app.services.report_service import (
    ReportGenerationError,
    ReportInputUnavailableError,
    ReportNotFoundError,
    ReportPersistenceError,
    ReportProviderExecutionError,
    ReportProviderRequiredError,
    ReportService,
)
from app.services.report_service_factory import build_configured_report_service
from app.success_notices import SuccessNotice, add_success_notice
from app.templating import templates


router = APIRouter(prefix="/incidents/{public_id}", tags=["reports"])
settings = get_settings()
DatabaseSession = Annotated[Session, Depends(get_db)]


def get_report_service(session: DatabaseSession) -> ReportService:
    """Return the report service used for scoped reads and human edits."""
    return ReportService(session)


def get_configured_report_service(session: DatabaseSession) -> ReportService:
    """Return the report service with the configured generation provider."""
    return build_configured_report_service(session, settings)


ReportServiceDependency = Annotated[ReportService, Depends(get_report_service)]
ConfiguredReportServiceDependency = Annotated[
    ReportService,
    Depends(get_configured_report_service),
]


def _report_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (ReportNotFoundError, AnalysisRunNotFoundError)):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(
        exc,
        (
            ReportGenerationError,
            ReportInputUnavailableError,
            ReportProviderRequiredError,
        ),
    ):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, ReportProviderExecutionError):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        )
    if isinstance(exc, ReportPersistenceError):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="The report request failed safely.",
    )


@router.post("/analysis/{run_id}/report", name="generate_report")
def generate_report(
    public_id: str,
    run_id: int,
    service: ConfiguredReportServiceDependency,
) -> RedirectResponse:
    """Generate once, or reopen the existing report for the scoped run."""
    try:
        report = service.generate_draft_report(public_id, run_id)
    except (
        AnalysisRunNotFoundError,
        ReportGenerationError,
        ReportInputUnavailableError,
        ReportPersistenceError,
        ReportProviderExecutionError,
        ReportProviderRequiredError,
    ) as exc:
        raise _report_error(exc) from exc
    location = add_success_notice(
        f"/incidents/{public_id}/reports/{report.id}",
        SuccessNotice.REPORT_DRAFT_READY,
    )
    return RedirectResponse(
        url=location,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/reports/{report_id}", name="report_detail")
def report_detail(
    request: Request,
    public_id: str,
    report_id: int,
    service: ReportServiceDependency,
) -> HTMLResponse:
    """Render one incident-scoped editable report draft."""
    try:
        report = service.get_report(public_id, report_id)
    except ReportNotFoundError as exc:
        raise _report_error(exc) from exc
    return templates.TemplateResponse(
        request=request,
        name="report.html",
        context={
            "app_name": settings.app_name,
            "report": report,
            "incident": report.incident,
            "analysis_run": report.analysis_run,
        },
    )


@router.post("/reports/{report_id}", name="save_report")
def save_report(
    public_id: str,
    report_id: int,
    service: ReportServiceDependency,
    editable_text: Annotated[str, Form()],
) -> RedirectResponse:
    """Save a human-edited draft while retaining the AI-generated original."""
    try:
        update = ReportDraftUpdate(editable_text=editable_text)
        service.save_report_edit(public_id, report_id, update)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=validation_messages(exc),
        ) from exc
    except (ReportNotFoundError, ReportPersistenceError) as exc:
        raise _report_error(exc) from exc
    location = add_success_notice(
        f"/incidents/{public_id}/reports/{report_id}",
        SuccessNotice.REPORT_DRAFT_UPDATED,
    )
    return RedirectResponse(
        url=location,
        status_code=status.HTTP_303_SEE_OTHER,
    )
