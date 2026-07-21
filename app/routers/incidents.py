"""Incident creation, listing, detail, and update routes."""

from datetime import UTC, datetime
from typing import Annotated
from zoneinfo import ZoneInfo

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
from app.success_notices import SuccessNotice, add_success_notice
from app.templating import templates


router = APIRouter(prefix="/incidents", tags=["incidents"])
settings = get_settings()
DatabaseSession = Annotated[Session, Depends(get_db)]


class _IncidentFormTimeError(ValueError):
    """Raised when a submitted local incident time cannot be interpreted."""


def _parse_reported_start_time(value: str) -> datetime | None:
    if not value:
        return None

    try:
        local_datetime = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _IncidentFormTimeError(
            "Reported start time must be a valid local date and time."
        ) from exc
    if local_datetime.tzinfo is not None:
        raise _IncidentFormTimeError(
            "Reported start time must not include a timezone offset."
        )

    display_timezone = ZoneInfo(settings.display_timezone)
    utc_datetime = local_datetime.replace(tzinfo=display_timezone).astimezone(UTC)
    round_trip = utc_datetime.astimezone(display_timezone).replace(tzinfo=None)
    if round_trip != local_datetime:
        raise _IncidentFormTimeError(
            f"Reported start time does not exist in {settings.display_timezone}."
        )
    return utc_datetime


def _incident_form_values(incident: Incident) -> dict[str, str]:
    reported_start_time = (
        incident.reported_start_time.astimezone(ZoneInfo(settings.display_timezone))
        .replace(tzinfo=None)
        .isoformat(timespec="minutes")
        if incident.reported_start_time is not None
        else ""
    )
    return {
        "name": incident.name,
        "description": incident.description,
        "affected_service": incident.affected_service,
        "reported_start_time": reported_start_time,
    }


def _incident_form_context(
    *,
    errors: list[str],
    values: dict[str, str],
    incident: Incident | None = None,
) -> dict[str, object]:
    context: dict[str, object] = {
        "app_name": settings.app_name,
        "display_timezone": settings.display_timezone,
        "errors": errors,
        "values": values,
    }
    if incident is not None:
        context["incident"] = incident
    return context


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
        context=_incident_form_context(errors=[], values={}),
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
            reported_start_time=_parse_reported_start_time(reported_start_time),
        )
        incident = IncidentService(session).create_incident(incident_data)
    except (_IncidentFormTimeError, ValidationError) as exc:
        errors = (
            [str(exc)]
            if isinstance(exc, _IncidentFormTimeError)
            else validation_messages(exc)
        )
        return templates.TemplateResponse(
            request=request,
            name="incident_form.html",
            context=_incident_form_context(errors=errors, values=values),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
    except IncidentPersistenceError as exc:
        return templates.TemplateResponse(
            request=request,
            name="incident_form.html",
            context=_incident_form_context(errors=[str(exc)], values=values),
            status_code=status.HTTP_409_CONFLICT,
        )

    return RedirectResponse(
        url=add_success_notice(
            f"/incidents/{incident.public_id}",
            SuccessNotice.INCIDENT_CREATED,
        ),
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
        context=_incident_form_context(
            incident=incident,
            errors=[],
            values=_incident_form_values(incident),
        ),
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
            reported_start_time=_parse_reported_start_time(reported_start_time),
        )
        incident = service.update_incident(public_id, incident_data)
    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (_IncidentFormTimeError, ValidationError) as exc:
        try:
            incident = service.get_incident_or_raise(public_id)
        except IncidentNotFoundError as missing_exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(missing_exc),
            ) from missing_exc
        errors = (
            [str(exc)]
            if isinstance(exc, _IncidentFormTimeError)
            else validation_messages(exc)
        )
        return templates.TemplateResponse(
            request=request,
            name="incident_detail.html",
            context=_incident_form_context(
                incident=incident,
                errors=errors,
                values=values,
            ),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
    except IncidentPersistenceError as exc:
        incident = service.get_incident_or_raise(public_id)
        return templates.TemplateResponse(
            request=request,
            name="incident_detail.html",
            context=_incident_form_context(
                incident=incident,
                errors=[str(exc)],
                values=values,
            ),
            status_code=status.HTTP_409_CONFLICT,
        )

    return RedirectResponse(
        url=add_success_notice(
            f"/incidents/{incident.public_id}",
            SuccessNotice.INCIDENT_UPDATED,
        ),
        status_code=status.HTTP_303_SEE_OTHER,
    )
