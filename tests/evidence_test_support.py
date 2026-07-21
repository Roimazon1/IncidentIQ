"""Small shared helpers for evidence-focused tests."""

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import EvidenceItem, Incident, IncidentStatus
from app.schemas.incident import IncidentCreate
from app.services.incident_service import IncidentService


def create_incident(
    session: Session,
    *,
    name: str = "Checkout failures",
) -> str:
    incident = IncidentService(session).create_incident(
        IncidentCreate(
            name=name,
            description="Intermittent checkout errors",
            affected_service="checkout",
        )
    )
    return incident.public_id


def create_incident_through_api(database_client: TestClient) -> None:
    response = database_client.post(
        "/incidents",
        data={
            "name": "Checkout failures",
            "description": "Intermittent checkout errors",
            "affected_service": "checkout",
            "reported_start_time": "",
        },
    )
    assert response.status_code == 200


def assert_no_evidence_and_draft(
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as session:
        evidence_count = session.scalar(
            select(func.count()).select_from(EvidenceItem)
        )
        incident = session.scalar(select(Incident))
    assert evidence_count == 0
    assert incident is not None
    assert incident.status is IncidentStatus.DRAFT
