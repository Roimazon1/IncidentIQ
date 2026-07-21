"""Dashboard route for saved incident investigations."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.services.incident_service import IncidentService
from app.templating import templates


router = APIRouter(tags=["dashboard"])
settings = get_settings()
DatabaseSession = Annotated[Session, Depends(get_db)]


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, session: DatabaseSession) -> HTMLResponse:
    """Render a bounded newest-first dashboard of saved incidents."""
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "app_name": settings.app_name,
            "incidents": IncidentService(session).list_incidents(),
        },
    )
