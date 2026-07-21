"""Persistence service for incident evidence ingestion."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import EvidenceItem, EvidenceType
from app.models.identifiers import EVIDENCE_CODE_PREFIX, generate_evidence_code
from app.schemas.evidence import EvidenceCreate
from app.services.incident_service import IncidentService


class EvidencePersistenceError(RuntimeError):
    """Raised when an evidence item cannot be persisted safely."""


SUPPORTED_UPLOAD_EXTENSIONS = (".txt", ".log", ".json", ".csv", ".md")


@dataclass(frozen=True, slots=True)
class EvidenceUpload:
    """One uploaded evidence file before deterministic validation and storage."""

    filename: str
    content: bytes


class IngestionService:
    """Create traceable evidence items within one database session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def ingest_pasted_text(
        self,
        incident_public_id: str,
        evidence_data: EvidenceCreate,
    ) -> EvidenceItem:
        """Persist original pasted text with a stable per-incident evidence code."""
        return self._persist_evidence_items(
            incident_public_id,
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
    ) -> list[EvidenceItem]:
        """Decode and persist an uploaded file batch in one transaction."""
        if not uploads:
            raise ValueError("at least one uploaded evidence file is required")

        evidence_data = [
            EvidenceCreate(
                source_name=upload.filename,
                evidence_type=EvidenceType.OTHER,
                original_text=upload.content.decode("utf-8"),
            )
            for upload in uploads
        ]
        return self._persist_evidence_items(
            incident_public_id,
            evidence_data,
            failure_message="The uploaded evidence could not be saved.",
        )

    def _persist_evidence_items(
        self,
        incident_public_id: str,
        evidence_data: Sequence[EvidenceCreate],
        *,
        failure_message: str,
    ) -> list[EvidenceItem]:
        incident_service = IncidentService(self.session)
        incident = incident_service.get_incident_or_raise(incident_public_id)
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
