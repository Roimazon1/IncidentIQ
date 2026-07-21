"""Evidence entry routes for pasted text and uploaded files."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import EvidenceType, Incident
from app.routers.validation import validation_messages
from app.schemas.evidence import EvidenceCreate
from app.services.incident_service import IncidentNotFoundError, IncidentService
from app.services.ingestion_service import (
    SUPPORTED_UPLOAD_EXTENSIONS,
    EvidencePersistenceError,
    EvidenceUpload,
    IngestionService,
)
from app.templating import templates


router = APIRouter(tags=["evidence"])
settings = get_settings()
DatabaseSession = Annotated[Session, Depends(get_db)]
UPLOAD_ACCEPT = ",".join(SUPPORTED_UPLOAD_EXTENSIONS)
UPLOAD_FORMATS_TEXT = (
    f"{', '.join(SUPPORTED_UPLOAD_EXTENSIONS[:-1])}, and "
    f"{SUPPORTED_UPLOAD_EXTENSIONS[-1]}"
)


def _evidence_form_context(
    incident: Incident,
    *,
    errors: list[str],
    values: dict[str, str],
) -> dict[str, object]:
    return {
        "app_name": settings.app_name,
        "incident": incident,
        "errors": errors,
        "values": values,
        "upload_accept": UPLOAD_ACCEPT,
        "upload_formats_text": UPLOAD_FORMATS_TEXT,
    }


@router.get(
    "/incidents/{public_id}/evidence/new",
    response_class=HTMLResponse,
)
def new_evidence_form(
    request: Request,
    public_id: str,
    session: DatabaseSession,
) -> HTMLResponse:
    """Render the pasted-text and file-upload form for one incident."""
    try:
        incident = IncidentService(session).get_incident_or_raise(public_id)
    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return templates.TemplateResponse(
        request=request,
        name="evidence_form.html",
        context=_evidence_form_context(
            incident,
            errors=[],
            values={"source_name": "Pasted text"},
        ),
    )


@router.post(
    "/incidents/{public_id}/evidence/text",
    response_class=HTMLResponse,
)
def create_pasted_evidence(
    request: Request,
    public_id: str,
    session: DatabaseSession,
    source_name: Annotated[str, Form()],
    original_text: Annotated[str, Form()],
) -> Response:
    """Validate and persist pasted text, then redirect to its incident."""
    service = IngestionService(session)
    values = {"source_name": source_name, "original_text": original_text}
    try:
        evidence_data = EvidenceCreate(
            source_name=source_name,
            evidence_type=EvidenceType.OTHER,
            original_text=original_text,
        )
        service.ingest_pasted_text(public_id, evidence_data)
    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValidationError as exc:
        try:
            incident = IncidentService(session).get_incident_or_raise(public_id)
        except IncidentNotFoundError as missing_exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(missing_exc),
            ) from missing_exc
        return templates.TemplateResponse(
            request=request,
            name="evidence_form.html",
            context=_evidence_form_context(
                incident,
                errors=validation_messages(exc),
                values=values,
            ),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
    except EvidencePersistenceError as exc:
        incident = IncidentService(session).get_incident_or_raise(public_id)
        return templates.TemplateResponse(
            request=request,
            name="evidence_form.html",
            context=_evidence_form_context(
                incident,
                errors=[str(exc)],
                values=values,
            ),
            status_code=status.HTTP_409_CONFLICT,
        )

    return RedirectResponse(
        url=f"/incidents/{public_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post(
    "/incidents/{public_id}/evidence/upload",
    response_class=HTMLResponse,
)
def create_uploaded_evidence(
    request: Request,
    public_id: str,
    session: DatabaseSession,
    files: Annotated[list[UploadFile], File()],
) -> Response:
    """Persist multiple uploaded evidence files, then redirect to the incident."""
    uploads = [
        EvidenceUpload(
            filename=uploaded_file.filename or "",
            content=uploaded_file.file.read(),
        )
        for uploaded_file in files
    ]
    try:
        IngestionService(session).ingest_uploaded_files(public_id, uploads)
    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except EvidencePersistenceError as exc:
        incident = IncidentService(session).get_incident_or_raise(public_id)
        return templates.TemplateResponse(
            request=request,
            name="evidence_form.html",
            context=_evidence_form_context(
                incident,
                errors=[str(exc)],
                values={"source_name": "Pasted text"},
            ),
            status_code=status.HTTP_409_CONFLICT,
        )

    return RedirectResponse(
        url=f"/incidents/{public_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )
