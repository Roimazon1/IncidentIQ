"""Focused tests for incident lifecycle and bounded listing rules."""

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.models import EvidenceItem, EvidenceType, Incident, IncidentStatus
from app.schemas.incident import IncidentCreate
from app.services.incident_service import (
    MAX_INCIDENT_LIST_LIMIT,
    IncidentService,
)


@pytest.fixture
def service_session(
    database_session_factory: sessionmaker[Session],
) -> Iterator[Session]:
    with database_session_factory() as session:
        yield session


def _create_incident(
    service: IncidentService,
    *,
    name: str = "Checkout failures",
) -> Incident:
    return service.create_incident(
        IncidentCreate(
            name=name,
            description="Intermittent checkout errors",
            affected_service="checkout",
        )
    )


def _evidence_item() -> EvidenceItem:
    return EvidenceItem(
        evidence_code="E-001",
        source_name="checkout.log",
        evidence_type=EvidenceType.APPLICATION_LOG,
        original_text="Checkout failed",
        checksum="a" * 64,
    )


def test_recalculate_status_without_evidence_produces_draft(
    service_session: Session,
) -> None:
    service = IncidentService(service_session)
    incident = _create_incident(service)
    incident.status = IncidentStatus.READY

    status = service.recalculate_status(incident)

    assert status is IncidentStatus.DRAFT
    assert incident.status is IncidentStatus.DRAFT


def test_recalculate_status_with_persisted_evidence_produces_ready(
    service_session: Session,
) -> None:
    service = IncidentService(service_session)
    incident = _create_incident(service)
    evidence = _evidence_item()
    incident.evidence_items.append(evidence)
    service_session.commit()

    status = service.recalculate_status(incident)

    assert status is IncidentStatus.READY
    assert incident.status is IncidentStatus.READY


def test_removing_final_evidence_returns_incident_to_draft(
    service_session: Session,
) -> None:
    service = IncidentService(service_session)
    incident = _create_incident(service)
    evidence = _evidence_item()
    incident.evidence_items.append(evidence)
    service_session.commit()
    service.recalculate_status(incident)

    service_session.delete(evidence)
    service_session.flush()
    status = service.recalculate_status(incident)

    assert status is IncidentStatus.DRAFT
    assert incident.status is IncidentStatus.DRAFT


@pytest.mark.parametrize(
    "preserved_status",
    [
        IncidentStatus.ANALYZING,
        IncidentStatus.COMPLETED,
        IncidentStatus.FAILED,
    ],
)
def test_recalculate_status_preserves_analysis_outcomes(
    service_session: Session,
    preserved_status: IncidentStatus,
) -> None:
    service = IncidentService(service_session)
    incident = _create_incident(service)
    incident.status = preserved_status

    status = service.recalculate_status(incident)

    assert status is preserved_status
    assert incident.status is preserved_status


@pytest.mark.parametrize("invalid_limit", [0, MAX_INCIDENT_LIST_LIMIT + 1])
def test_list_incidents_rejects_invalid_limit(
    service_session: Session,
    invalid_limit: int,
) -> None:
    with pytest.raises(ValueError, match="limit must be between"):
        IncidentService(service_session).list_incidents(limit=invalid_limit)


def test_list_incidents_rejects_negative_offset(
    service_session: Session,
) -> None:
    with pytest.raises(ValueError, match="offset must not be negative"):
        IncidentService(service_session).list_incidents(offset=-1)


def test_list_incidents_is_newest_first_and_respects_limit_and_offset(
    service_session: Session,
) -> None:
    service = IncidentService(service_session)
    oldest = _create_incident(service, name="Oldest")
    middle = _create_incident(service, name="Middle")
    newest = _create_incident(service, name="Newest")
    oldest.created_at = datetime(2025, 1, 1, tzinfo=UTC)
    middle.created_at = datetime(2025, 1, 2, tzinfo=UTC)
    newest.created_at = datetime(2025, 1, 3, tzinfo=UTC)
    service_session.commit()

    incidents = service.list_incidents(limit=2, offset=1)

    assert [incident.name for incident in incidents] == ["Middle", "Oldest"]
