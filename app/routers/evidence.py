"""Evidence entry routes for pasted text and uploaded files."""

from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
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
from app.schemas.evidence import (
    EvidenceCreate,
    EvidenceManifestSource,
    EvidenceUpdate,
)
from app.services.evidence_manifest_service import (
    EvidenceManifestService,
    EvidencePreviewValidationError,
)
from app.services.incident_service import IncidentNotFoundError, IncidentService
from app.services.ingestion_service import (
    SUPPORTED_UPLOAD_EXTENSIONS,
    EvidenceItemNotFoundError,
    EvidencePersistenceError,
    EvidenceUpload,
    EvidenceUploadValidationError,
    IngestionService,
)
from app.templating import templates


router = APIRouter(tags=["evidence"])
settings = get_settings()
DatabaseSession = Annotated[Session, Depends(get_db)]
EvidenceFormTab = Literal["paste", "upload", "saved"]
UPLOAD_ACCEPT = ",".join(SUPPORTED_UPLOAD_EXTENSIONS)
UPLOAD_FORMATS_TEXT = (
    f"{', '.join(SUPPORTED_UPLOAD_EXTENSIONS[:-1])}, and "
    f"{SUPPORTED_UPLOAD_EXTENSIONS[-1]}"
)


def _evidence_form_context(
    session: Session,
    incident: Incident,
    *,
    errors: list[str],
    values: dict[str, str] | None = None,
    active_tab: EvidenceFormTab = "paste",
) -> dict[str, object]:
    form_values = {
        "source_name": "Pasted text",
        "pasted_evidence_type": EvidenceType.OTHER.value,
        "upload_evidence_type": EvidenceType.OTHER.value,
    }
    if values is not None:
        form_values.update(values)
    return {
        "app_name": settings.app_name,
        "incident": incident,
        "evidence_items": IngestionService(session).list_evidence_metadata(incident.id),
        "evidence_types": list(EvidenceType),
        "errors": errors,
        "values": form_values,
        "active_tab": active_tab,
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
    tab: Annotated[EvidenceFormTab, Query()] = "paste",
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
            session,
            incident,
            errors=[],
            active_tab=tab,
        ),
    )


@router.get(
    "/incidents/{public_id}/evidence/{evidence_code}",
    response_class=HTMLResponse,
)
def evidence_preview(
    request: Request,
    public_id: str,
    evidence_code: str,
    session: DatabaseSession,
) -> HTMLResponse:
    """Render one saved evidence item and its original local content."""
    try:
        evidence = IngestionService(session).get_evidence_or_raise(
            public_id,
            evidence_code,
        )
    except EvidenceItemNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return templates.TemplateResponse(
        request=request,
        name="evidence_preview.html",
        context={
            "app_name": settings.app_name,
            "incident": evidence.incident,
            "evidence": evidence,
        },
    )


@router.get(
    "/incidents/{public_id}/evidence/{evidence_code}/redaction-preview",
    response_class=HTMLResponse,
)
def redaction_preview(
    request: Request,
    public_id: str,
    evidence_code: str,
    session: DatabaseSession,
) -> HTMLResponse:
    """Render the outbound-safe redaction preview for one evidence item."""
    try:
        evidence = IngestionService(session).get_evidence_or_raise(
            public_id,
            evidence_code,
        )
        preview = EvidenceManifestService.build_redaction_preview(
            EvidenceManifestSource(
                evidence_code=evidence.evidence_code,
                source_name=evidence.source_name,
                evidence_type=evidence.evidence_type,
                original_text=evidence.original_text,
            )
        )
    except EvidenceItemNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except EvidencePreviewValidationError as exc:
        return templates.TemplateResponse(
            request=request,
            name="redaction_preview_error.html",
            context={
                "app_name": settings.app_name,
                "incident": evidence.incident,
                "evidence_code": evidence.evidence_code,
                "error_message": str(exc),
            },
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    return templates.TemplateResponse(
        request=request,
        name="redaction_preview.html",
        context={
            "app_name": settings.app_name,
            "incident": evidence.incident,
            "preview": preview,
        },
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
    evidence_type: Annotated[str, Form()] = EvidenceType.OTHER.value,
) -> Response:
    """Validate and persist pasted text, then redirect to its incident."""
    service = IngestionService(session)
    values = {
        "source_name": source_name,
        "original_text": original_text,
        "pasted_evidence_type": evidence_type,
    }
    try:
        evidence_data = EvidenceCreate(
            source_name=source_name,
            evidence_type=evidence_type,
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
                session,
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
                session,
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
    evidence_type: Annotated[str, Form()] = EvidenceType.OTHER.value,
) -> Response:
    """Validate and persist uploaded evidence, then redirect to the incident."""
    values = {"upload_evidence_type": evidence_type}
    try:
        evidence_update = EvidenceUpdate(evidence_type=evidence_type)
        uploads = [
            EvidenceUpload(
                filename=uploaded_file.filename or "",
                content=uploaded_file.file.read(settings.max_upload_bytes + 1),
                evidence_type=evidence_update.evidence_type,
            )
            for uploaded_file in files
        ]
        IngestionService(
            session,
            max_upload_bytes=settings.max_upload_bytes,
        ).ingest_uploaded_files(public_id, uploads)
    except IncidentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (EvidenceUploadValidationError, ValidationError) as exc:
        incident = IncidentService(session).get_incident_or_raise(public_id)
        errors = (
            [str(exc)]
            if isinstance(exc, EvidenceUploadValidationError)
            else validation_messages(exc)
        )
        return templates.TemplateResponse(
            request=request,
            name="evidence_form.html",
            context=_evidence_form_context(
                session,
                incident,
                errors=errors,
                values=values,
                active_tab="upload",
            ),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
    except EvidencePersistenceError as exc:
        incident = IncidentService(session).get_incident_or_raise(public_id)
        return templates.TemplateResponse(
            request=request,
            name="evidence_form.html",
            context=_evidence_form_context(
                session,
                incident,
                errors=[str(exc)],
                values=values,
                active_tab="upload",
            ),
            status_code=status.HTTP_409_CONFLICT,
        )

    return RedirectResponse(
        url=f"/incidents/{public_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post(
    "/incidents/{public_id}/evidence/{evidence_code}/type",
    response_class=HTMLResponse,
)
def update_evidence_type(
    request: Request,
    public_id: str,
    evidence_code: str,
    session: DatabaseSession,
    evidence_type: Annotated[str, Form()],
) -> Response:
    """Validate and persist a human correction to an evidence classification."""
    try:
        evidence_data = EvidenceUpdate(evidence_type=evidence_type)
        IngestionService(session).update_evidence_metadata(
            public_id,
            evidence_code,
            evidence_data,
        )
    except EvidenceItemNotFoundError as exc:
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
                session,
                incident,
                errors=validation_messages(exc),
                active_tab="saved",
            ),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
    except EvidencePersistenceError as exc:
        incident = IncidentService(session).get_incident_or_raise(public_id)
        return templates.TemplateResponse(
            request=request,
            name="evidence_form.html",
            context=_evidence_form_context(
                session,
                incident,
                errors=[str(exc)],
                active_tab="saved",
            ),
            status_code=status.HTTP_409_CONFLICT,
        )

    return RedirectResponse(
        url=f"/incidents/{public_id}/evidence/new?tab=saved",
        status_code=status.HTTP_303_SEE_OTHER,
    )
