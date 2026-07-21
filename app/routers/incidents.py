"""Incident creation, listing, detail, and update routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Incident
from app.routers.validation import validation_messages
from app.schemas.incident import IncidentCreate, IncidentRead, IncidentUpdate
from app.services.incident_service import (
    DEFAULT_INCIDENT_LIST_LIMIT,
    MAX_INCIDENT_LIST_LIMIT,
    IncidentNotFoundError,
    IncidentPersistenceError,
    IncidentService,
)
from app.templating import templates


router = APIRouter(prefix="/incidents", tags=["incidents"])
settings = get_settings()
DatabaseSession = Annotated[Session, Depends(get_db)]


def _incident_form_values(incident: Incident) -> dict[str, str]:
    reported_start_time = (
        incident.reported_start_time.isoformat()
        if incident.reported_start_time is not None
        else ""
    )
    return {
        "name": incident.name,
        "description": incident.description,
        "affected_service": incident.affected_service,
        "reported_start_time": reported_start_time,
    }


@router.get("", response_model=list[IncidentRead])
def list_incidents(
    session: DatabaseSession,
    limit: Annotated[
        int,
        Query(ge=1, le=MAX_INCIDENT_LIST_LIMIT),
    ] = DEFAULT_INCIDENT_LIST_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Incident]:
    """Return a bounded JSON page of persisted incidents."""
    return IncidentService(session).list_incidents(limit=limit, offset=offset)


@router.get("/new", response_class=HTMLResponse)
def new_incident_form(request: Request) -> HTMLResponse:
    """Render an empty incident metadata form."""
    return templates.TemplateResponse(
        request=request,
        name="incident_form.html",
        context={
            "app_name": settings.app_name,
            "errors": [],
            "values": {},
        },
    )


@router.post("", response_class=HTMLResponse)
def create_incident(
    request: Request,
    session: DatabaseSession,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    affected_service: Annotated[str, Form()],
    reported_start_time: Annotated[str, Form()] = "",
) -> Response:
    """Validate an incident form, persist it, and redirect to its detail page."""
    values = {
        "name": name,
        "description": description,
        "affected_service": affected_service,
        "reported_start_time": reported_start_time,
    }
    try:
        incident_data = IncidentCreate(
            name=name,
            description=description,
            affected_service=affected_service,
            reported_start_time=reported_start_time or None,
        )
        incident = IncidentService(session).create_incident(incident_data)
    except ValidationError as exc:
        return templates.TemplateResponse(
            request=request,
            name="incident_form.html",
            context={
                "app_name": settings.app_name,
                "errors": validation_messages(exc),
                "values": values,
            },
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
    except IncidentPersistenceError as exc:
        return templates.TemplateResponse(
            request=request,
            name="incident_form.html",
            context={
                "app_name": settings.app_name,
                "errors": [str(exc)],
                "values": values,
            },
            status_code=status.HTTP_409_CONFLICT,
        )

    return RedirectResponse(
        url=f"/incidents/{incident.public_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/{public_id}", response_class=HTMLResponse)
def incident_detail(
    request: Request,
    public_id: str,
    session: DatabaseSession,
) -> HTMLResponse:
    """Render one persisted incident without loading unrelated collections."""
    try:
        incident = IncidentService(session).get_incident_or_raise(public_id)
    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return templates.TemplateResponse(
        request=request,
        name="incident_detail.html",
        context={
            "app_name": settings.app_name,
            "incident": incident,
            "errors": [],
            "values": _incident_form_values(incident),
        },
    )


@router.post("/{public_id}", response_class=HTMLResponse)
def update_incident(
    request: Request,
    public_id: str,
    session: DatabaseSession,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    affected_service: Annotated[str, Form()],
    reported_start_time: Annotated[str, Form()] = "",
) -> Response:
    """Validate and persist user-editable incident metadata."""
    values = {
        "name": name,
        "description": description,
        "affected_service": affected_service,
        "reported_start_time": reported_start_time,
    }
    service = IncidentService(session)
    try:
        incident_data = IncidentUpdate(
            name=name,
            description=description,
            affected_service=affected_service,
            reported_start_time=reported_start_time or None,
        )
        incident = service.update_incident(public_id, incident_data)
    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValidationError as exc:
        try:
            incident = service.get_incident_or_raise(public_id)
        except IncidentNotFoundError as missing_exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(missing_exc),
            ) from missing_exc
        return templates.TemplateResponse(
            request=request,
            name="incident_detail.html",
            context={
                "app_name": settings.app_name,
                "incident": incident,
                "errors": validation_messages(exc),
                "values": values,
            },
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
    except IncidentPersistenceError as exc:
        incident = service.get_incident_or_raise(public_id)
        return templates.TemplateResponse(
            request=request,
            name="incident_detail.html",
            context={
                "app_name": settings.app_name,
                "incident": incident,
                "errors": [str(exc)],
                "values": values,
            },
            status_code=status.HTTP_409_CONFLICT,
        )

    return RedirectResponse(
        url=f"/incidents/{incident.public_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )
