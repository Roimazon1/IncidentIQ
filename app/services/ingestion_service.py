"""Persistence service for incident evidence ingestion."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePath

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload, load_only

from app.config import get_settings
from app.models import EvidenceItem, EvidenceType, Incident
from app.models.evidence import SOURCE_NAME_MAX_LENGTH
from app.models.identifiers import EVIDENCE_CODE_PREFIX, generate_evidence_code
from app.schemas.evidence import EvidenceCreate, EvidenceUpdate
from app.services.incident_service import IncidentService


class EvidencePersistenceError(RuntimeError):
    """Raised when an evidence item cannot be persisted safely."""


class EvidenceUploadValidationError(ValueError):
    """Raised when an uploaded file is unsafe or unsupported."""


class EvidenceItemNotFoundError(LookupError):
    """Raised when an evidence code does not belong to an incident."""


SUPPORTED_UPLOAD_EXTENSIONS = (".txt", ".log", ".json", ".csv", ".md")
READABLE_TEXT_CONTROLS = {"\t", "\n", "\r"}


@dataclass(frozen=True, slots=True)
class EvidenceUpload:
    """One uploaded evidence file before deterministic validation and storage."""

    filename: str
    content: bytes
    evidence_type: EvidenceType = EvidenceType.OTHER


class IngestionService:
    """Create traceable evidence items within one database session."""

    def __init__(
        self,
        session: Session,
        *,
        max_upload_bytes: int | None = None,
    ) -> None:
        self.session = session
        self.max_upload_bytes = (
            get_settings().max_upload_bytes
            if max_upload_bytes is None
            else max_upload_bytes
        )

    def ingest_pasted_text(
        self,
        incident_public_id: str,
        evidence_data: EvidenceCreate,
    ) -> EvidenceItem:
        """Persist original pasted text with a stable per-incident evidence code."""
        incident = IncidentService(self.session).get_incident_or_raise(
            incident_public_id
        )
        return self._persist_evidence_items(
            incident,
            [evidence_data],
            failure_message="The pasted evidence could not be saved.",
        )[0]

    def ingest_uploaded_file(
        self,
        incident_public_id: str,
        upload: EvidenceUpload,
    ) -> EvidenceItem:
        """Decode and persist one uploaded evidence file."""
        return self.ingest_uploaded_files(incident_public_id, [upload])[0]

    def ingest_uploaded_files(
        self,
        incident_public_id: str,
        uploads: Sequence[EvidenceUpload],
        *,
        commit: bool = True,
    ) -> list[EvidenceItem]:
        """Decode and persist uploads with an optional caller-managed transaction."""
        if not uploads:
            raise EvidenceUploadValidationError(
                "At least one uploaded evidence file is required."
            )

        incident = IncidentService(self.session).get_incident_or_raise(
            incident_public_id
        )

        evidence_data = [self._prepare_upload(upload) for upload in uploads]
        return self._persist_evidence_items(
            incident,
            evidence_data,
            failure_message="The uploaded evidence could not be saved.",
            commit=commit,
        )

    def list_evidence_metadata(self, incident_id: int) -> list[EvidenceItem]:
        """Return saved evidence identifiers and editable metadata only."""
        statement = (
            select(EvidenceItem)
            .options(
                load_only(
                    EvidenceItem.id,
                    EvidenceItem.incident_id,
                    EvidenceItem.evidence_code,
                    EvidenceItem.source_name,
                    EvidenceItem.evidence_type,
                )
            )
            .where(EvidenceItem.incident_id == incident_id)
            .order_by(EvidenceItem.id)
        )
        return list(self.session.scalars(statement))

    def get_evidence_or_raise(
        self,
        incident_public_id: str,
        evidence_code: str,
    ) -> EvidenceItem:
        """Return one full evidence item scoped to its incident."""
        evidence = self.session.scalar(
            select(EvidenceItem)
            .join(Incident)
            .options(joinedload(EvidenceItem.incident))
            .where(
                Incident.public_id == incident_public_id,
                EvidenceItem.evidence_code == evidence_code,
            )
        )
        if evidence is None:
            raise EvidenceItemNotFoundError(
                f"Evidence {evidence_code} was not found for incident "
                f"{incident_public_id}."
            )
        return evidence

    def update_evidence_metadata(
        self,
        incident_public_id: str,
        evidence_code: str,
        evidence_data: EvidenceUpdate,
    ) -> EvidenceItem:
        """Persist user-correctable metadata for one incident evidence item."""
        evidence = self.session.scalar(
            select(EvidenceItem)
            .join(Incident)
            .options(
                load_only(
                    EvidenceItem.id,
                    EvidenceItem.incident_id,
                    EvidenceItem.evidence_code,
                    EvidenceItem.source_name,
                    EvidenceItem.evidence_type,
                )
            )
            .where(
                Incident.public_id == incident_public_id,
                EvidenceItem.evidence_code == evidence_code,
            )
        )
        if evidence is None:
            raise EvidenceItemNotFoundError(
                f"Evidence {evidence_code} was not found for incident "
                f"{incident_public_id}."
            )

        for field_name, value in evidence_data.model_dump(exclude_unset=True).items():
            setattr(evidence, field_name, value)
        try:
            self.session.commit()
        except SQLAlchemyError as exc:
            self.session.rollback()
            raise EvidencePersistenceError(
                f"Evidence {evidence_code} could not be updated."
            ) from exc
        return evidence

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Return a storage-safe basename for an uploaded source label."""
        basename = PurePath(filename.replace("\\", "/")).name.strip()
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
        if not sanitized:
            raise EvidenceUploadValidationError(
                "Uploaded evidence must include a valid filename."
            )
        if len(sanitized) > SOURCE_NAME_MAX_LENGTH:
            raise EvidenceUploadValidationError(
                "Uploaded filenames must be "
                f"{SOURCE_NAME_MAX_LENGTH} characters or fewer."
            )
        return sanitized

    @staticmethod
    def validate_extension(filename: str) -> None:
        """Reject filenames outside the locked text-extension allowlist."""
        extension = PurePath(filename).suffix.lower()
        if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
            supported = ", ".join(SUPPORTED_UPLOAD_EXTENSIONS)
            raise EvidenceUploadValidationError(
                f"{filename} has an unsupported extension. "
                f"Supported extensions: {supported}."
            )

    def validate_size(self, filename: str, content: bytes) -> None:
        """Reject content larger than the configured per-file byte limit."""
        if len(content) > self.max_upload_bytes:
            raise EvidenceUploadValidationError(
                f"{filename} exceeds the maximum upload size of "
                f"{self.max_upload_bytes} bytes."
            )

    @staticmethod
    def decode_text(filename: str, content: bytes) -> str:
        """Decode UTF-8 evidence and reject unreadable binary controls."""
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvidenceUploadValidationError(
                f"{filename} must contain valid UTF-8 text."
            ) from exc

        contains_binary_control = any(
            ord(character) < 32 and character not in READABLE_TEXT_CONTROLS
            for character in decoded
        )
        if contains_binary_control:
            raise EvidenceUploadValidationError(
                f"{filename} contains unreadable binary content."
            )
        return decoded

    def _prepare_upload(self, upload: EvidenceUpload) -> EvidenceCreate:
        sanitized = self.sanitize_filename(upload.filename)
        self.validate_extension(sanitized)
        self.validate_size(sanitized, upload.content)
        return EvidenceCreate(
            source_name=sanitized,
            evidence_type=upload.evidence_type,
            original_text=self.decode_text(sanitized, upload.content),
        )

    def _persist_evidence_items(
        self,
        incident: Incident,
        evidence_data: Sequence[EvidenceCreate],
        *,
        failure_message: str,
        commit: bool = True,
    ) -> list[EvidenceItem]:
        incident_service = IncidentService(self.session)
        first_sequence = self._next_evidence_sequence(incident.id)
        evidence_items = [
            EvidenceItem(
                incident=incident,
                evidence_code=generate_evidence_code(first_sequence + offset),
                source_name=item.source_name,
                evidence_type=item.evidence_type,
                original_text=item.original_text,
                checksum=self.calculate_checksum(item.original_text),
            )
            for offset, item in enumerate(evidence_data)
        ]
        self.session.add_all(evidence_items)

        try:
            self.session.flush()
            incident_service.recalculate_status(incident)
            if commit:
                self.session.commit()
        except SQLAlchemyError as exc:
            self.session.rollback()
            raise EvidencePersistenceError(failure_message) from exc
        return evidence_items

    @staticmethod
    def calculate_checksum(content: str) -> str:
        """Return the lowercase SHA-256 checksum of the exact UTF-8 content."""
        return sha256(content.encode("utf-8")).hexdigest()

    def _next_evidence_sequence(self, incident_id: int) -> int:
        latest_code = self.session.scalar(
            select(func.max(EvidenceItem.evidence_code)).where(
                EvidenceItem.incident_id == incident_id
            )
        )
        return (
            1
            if latest_code is None
            else int(latest_code.removeprefix(EVIDENCE_CODE_PREFIX)) + 1
        )
