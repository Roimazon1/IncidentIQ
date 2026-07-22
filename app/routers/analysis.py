"""Analysis-run orchestration and saved-result page routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import AnalysisRunStatus
from app.services.analysis_service import (
    AnalysisAlreadyRunningError,
    AnalysisEvidenceRequiredError,
    AnalysisPersistenceError,
    AnalysisProviderRequiredError,
    AnalysisRunNotFoundError,
    AnalysisRunTransitionError,
    AnalysisService,
    build_configured_analysis_service,
)
from app.services.incident_service import IncidentNotFoundError
from app.templating import templates


router = APIRouter(prefix="/incidents/{public_id}/analysis", tags=["analysis"])
settings = get_settings()
DatabaseSession = Annotated[Session, Depends(get_db)]


def get_analysis_service(session: DatabaseSession) -> AnalysisService:
    """Return a read-only-capable service without constructing a provider."""
    return AnalysisService(session)


def get_configured_analysis_service(session: DatabaseSession) -> AnalysisService:
    """Return the settings-selected service used to execute a new run."""
    return build_configured_analysis_service(session, settings)


AnalysisServiceDependency = Annotated[AnalysisService, Depends(get_analysis_service)]
ConfiguredAnalysisServiceDependency = Annotated[
    AnalysisService,
    Depends(get_configured_analysis_service),
]


@router.post("", response_class=HTMLResponse)
def start_analysis(
    public_id: str,
    service: ConfiguredAnalysisServiceDependency,
) -> Response:
    """Run the core analysis pipeline and redirect to its retained result."""
    try:
        analysis_run = service.start_configured_analysis_run(public_id)
        analysis_run = service.run_core_analysis_to_terminal(analysis_run.id)
    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (
        AnalysisAlreadyRunningError,
        AnalysisEvidenceRequiredError,
        AnalysisPersistenceError,
        AnalysisProviderRequiredError,
        AnalysisRunTransitionError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return RedirectResponse(
        url=f"/incidents/{public_id}/analysis/{analysis_run.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/{run_id}", response_class=HTMLResponse)
def analysis_detail(
    request: Request,
    public_id: str,
    run_id: int,
    service: AnalysisServiceDependency,
) -> HTMLResponse:
    """Render one incident-scoped running, completed, or failed analysis."""
    try:
        page = service.get_analysis_page_data(public_id, run_id)
    except AnalysisRunNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    template_name = (
        "analysis_pending.html"
        if page.analysis_run.status is AnalysisRunStatus.RUNNING
        else "analysis.html"
    )
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context={
            "app_name": settings.app_name,
            "page": page,
            "incident": page.analysis_run.incident,
            "analysis_run": page.analysis_run,
        },
    )
