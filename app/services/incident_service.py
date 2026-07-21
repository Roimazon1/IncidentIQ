"""Incident persistence operations and lifecycle rules."""

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import EvidenceItem, Incident, IncidentStatus
from app.models.identifiers import (
    INCIDENT_PUBLIC_ID_PREFIX,
    generate_incident_public_id,
)
from app.schemas.incident import IncidentCreate, IncidentUpdate


DEFAULT_INCIDENT_LIST_LIMIT = 100
MAX_INCIDENT_LIST_LIMIT = 100


class IncidentNotFoundError(LookupError):
    """Raised when an incident public ID does not exist."""


class IncidentPersistenceError(RuntimeError):
    """Raised when an incident write cannot be completed safely."""


class IncidentService:
    """Create, retrieve, list, and update incidents in one session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_incident(self, incident_data: IncidentCreate) -> Incident:
        """Persist a new draft incident with the next deterministic public ID."""
        incident = Incident(
            public_id=self._next_public_id(),
            **incident_data.model_dump(),
        )
        self.session.add(incident)
        try:
            self.session.commit()
            self.session.refresh(incident)
        except SQLAlchemyError as exc:
            self.session.rollback()
            raise IncidentPersistenceError(
                "The incident could not be created."
            ) from exc
        return incident

    def get_incident_or_raise(self, public_id: str) -> Incident:
        """Return one incident or raise a domain-specific missing-resource error."""
        incident = self.session.scalar(
            select(Incident).where(Incident.public_id == public_id)
        )
        if incident is None:
            raise IncidentNotFoundError(f"Incident {public_id} was not found.")
        return incident

    def list_incidents(
        self,
        *,
        limit: int = DEFAULT_INCIDENT_LIST_LIMIT,
        offset: int = 0,
    ) -> list[Incident]:
        """Return a bounded, newest-first page of incidents."""
        if not 1 <= limit <= MAX_INCIDENT_LIST_LIMIT:
            raise ValueError(
                f"limit must be between 1 and {MAX_INCIDENT_LIST_LIMIT}"
            )
        if offset < 0:
            raise ValueError("offset must not be negative")

        statement = (
            select(Incident)
            .order_by(Incident.created_at.desc(), Incident.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(statement))

    def update_incident(
        self,
        public_id: str,
        incident_data: IncidentUpdate,
    ) -> Incident:
        """Apply a partial incident update and persist its recalculated status."""
        incident = self.get_incident_or_raise(public_id)
        for field_name, value in incident_data.model_dump(
            exclude_unset=True
        ).items():
            setattr(incident, field_name, value)
        self.recalculate_status(incident)

        try:
            self.session.commit()
            self.session.refresh(incident)
        except SQLAlchemyError as exc:
            self.session.rollback()
            raise IncidentPersistenceError(
                f"Incident {public_id} could not be updated."
            ) from exc
        return incident

    def recalculate_status(self, incident: Incident) -> IncidentStatus:
        """Set DRAFT/READY from evidence while preserving analysis outcomes."""
        if incident.status in {
            IncidentStatus.ANALYZING,
            IncidentStatus.COMPLETED,
            IncidentStatus.FAILED,
        }:
            return incident.status

        if incident.id is None:
            has_evidence = bool(incident.evidence_items)
        else:
            has_evidence = (
                self.session.scalar(
                    select(EvidenceItem.id)
                    .where(EvidenceItem.incident_id == incident.id)
                    .limit(1)
                )
                is not None
            )
        incident.status = (
            IncidentStatus.READY if has_evidence else IncidentStatus.DRAFT
        )
        return incident.status

    def _next_public_id(self) -> str:
        latest_public_id = self.session.scalar(select(func.max(Incident.public_id)))
        next_sequence = (
            1
            if latest_public_id is None
            else int(latest_public_id.removeprefix(INCIDENT_PUBLIC_ID_PREFIX)) + 1
        )
        return generate_incident_public_id(next_sequence)
