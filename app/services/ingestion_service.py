"""Persistence service for incident evidence ingestion."""

from hashlib import sha256

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import EvidenceItem
from app.models.identifiers import EVIDENCE_CODE_PREFIX, generate_evidence_code
from app.schemas.evidence import EvidenceCreate
from app.services.incident_service import IncidentService


class EvidencePersistenceError(RuntimeError):
    """Raised when an evidence item cannot be persisted safely."""


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
        incident_service = IncidentService(self.session)
        incident = incident_service.get_incident_or_raise(incident_public_id)
        evidence = EvidenceItem(
            incident=incident,
            evidence_code=self._next_evidence_code(incident.id),
            source_name=evidence_data.source_name,
            evidence_type=evidence_data.evidence_type,
            original_text=evidence_data.original_text,
            checksum=self.calculate_checksum(evidence_data.original_text),
        )
        self.session.add(evidence)

        try:
            self.session.flush()
            incident_service.recalculate_status(incident)
            self.session.commit()
            self.session.refresh(evidence)
        except SQLAlchemyError as exc:
            self.session.rollback()
            raise EvidencePersistenceError(
                "The pasted evidence could not be saved."
            ) from exc
        return evidence

    @staticmethod
    def calculate_checksum(content: str) -> str:
        """Return the lowercase SHA-256 checksum of the exact UTF-8 content."""
        return sha256(content.encode("utf-8")).hexdigest()

    def _next_evidence_code(self, incident_id: int) -> str:
        latest_code = self.session.scalar(
            select(func.max(EvidenceItem.evidence_code)).where(
                EvidenceItem.incident_id == incident_id
            )
        )
        next_sequence = (
            1
            if latest_code is None
            else int(latest_code.removeprefix(EVIDENCE_CODE_PREFIX)) + 1
        )
        return generate_evidence_code(next_sequence)
