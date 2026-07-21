"""Focused tests for pasted-text evidence creation."""

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import EvidenceItem, EvidenceType, Incident, IncidentStatus
from app.schemas.evidence import EvidenceCreate
from app.schemas.incident import IncidentCreate
from app.services.incident_service import IncidentService
from app.services.ingestion_service import IngestionService


def _create_incident(session: Session, *, name: str = "Checkout failures") -> str:
    incident = IncidentService(session).create_incident(
        IncidentCreate(
            name=name,
            description="Intermittent checkout errors",
            affected_service="checkout",
        )
    )
    return incident.public_id


def test_calculate_checksum_uses_exact_utf8_content() -> None:
    assert IngestionService.calculate_checksum("hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_pasted_text_is_preserved_with_sequential_per_incident_codes(
    database_session_factory: sessionmaker[Session],
) -> None:
    original_text = "  2025-01-01 ERROR checkout failed\n"

    with database_session_factory() as session:
        first_public_id = _create_incident(session)
        second_public_id = _create_incident(session, name="Payment failures")
        service = IngestionService(session)
        first = service.ingest_pasted_text(
            first_public_id,
            EvidenceCreate(
                source_name="Pasted application log",
                evidence_type=EvidenceType.OTHER,
                original_text=original_text,
            ),
        )
        second = service.ingest_pasted_text(
            first_public_id,
            EvidenceCreate(
                source_name="Pasted support note",
                evidence_type=EvidenceType.OTHER,
                original_text="Customer reported checkout errors",
            ),
        )
        other_incident_evidence = service.ingest_pasted_text(
            second_public_id,
            EvidenceCreate(
                source_name="Pasted payment note",
                evidence_type=EvidenceType.OTHER,
                original_text="Payment request timed out",
            ),
        )

        assert first.evidence_code == "E-001"
        assert second.evidence_code == "E-002"
        assert other_incident_evidence.evidence_code == "E-001"

    with database_session_factory() as session:
        saved_evidence = session.scalar(
            select(EvidenceItem).where(EvidenceItem.id == first.id)
        )
        assert saved_evidence is not None
        assert saved_evidence.original_text == original_text
        assert saved_evidence.redacted_text is None
        assert saved_evidence.checksum == (
            "209a2c6b609123f500e87398a750503068035e95504da3a14a1d500fc55f096b"
        )
        assert saved_evidence.incident.status is IncidentStatus.READY


def test_pasted_evidence_form_and_post_redirect_flow(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    database_client.post(
        "/incidents",
        data={
            "name": "Checkout failures",
            "description": "Intermittent checkout errors",
            "affected_service": "checkout",
            "reported_start_time": "",
        },
    )

    form_response = database_client.get(
        "/incidents/INC-000001/evidence/new"
    )
    assert form_response.status_code == 200
    assert "Add pasted evidence" in form_response.text
    assert "Evidence text" in form_response.text
    assert "type=\"file\"" not in form_response.text
    assert "Evidence type" not in form_response.text

    create_response = database_client.post(
        "/incidents/INC-000001/evidence/text",
        data={
            "source_name": "Pasted checkout log",
            "original_text": "ERROR checkout failed",
        },
        follow_redirects=False,
    )
    assert create_response.status_code == 303
    assert create_response.headers["location"] == "/incidents/INC-000001"

    with database_session_factory() as session:
        saved_evidence = session.scalar(select(EvidenceItem))
        assert saved_evidence is not None
        assert saved_evidence.evidence_code == "E-001"
        assert saved_evidence.evidence_type is EvidenceType.OTHER
        assert saved_evidence.incident.status is IncidentStatus.READY


def test_blank_pasted_evidence_preserves_form_and_creates_nothing(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    database_client.post(
        "/incidents",
        data={
            "name": "Checkout incident",
            "description": "Checkout errors after deployment.",
            "affected_service": "checkout-api",
            "reported_start_time": "",
        },
    )
    source_name = "  Operator notes  "
    original_text = " \n\t "

    response = database_client.post(
        "/incidents/INC-000001/evidence/text",
        data={"source_name": source_name, "original_text": original_text},
    )

    assert response.status_code == 422
    assert "evidence text must not be blank" in response.text
    assert f'value="{source_name}"' in response.text
    assert f">{original_text}</textarea>" in response.text
    with database_session_factory() as session:
        evidence_count = session.scalar(
            select(func.count()).select_from(EvidenceItem)
        )
        incident = session.scalar(select(Incident))
    assert evidence_count == 0
    assert incident is not None
    assert incident.status is IncidentStatus.DRAFT


def test_get_pasted_evidence_form_for_missing_incident_returns_not_found(
    database_client: TestClient,
) -> None:
    response = database_client.get("/incidents/INC-999999/evidence/new")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Incident INC-999999 was not found."
    }


def test_posting_pasted_evidence_to_missing_incident_returns_not_found(
    database_client: TestClient,
) -> None:
    response = database_client.post(
        "/incidents/INC-999999/evidence/text",
        data={
            "source_name": "Pasted text",
            "original_text": "Checkout failed",
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Incident INC-999999 was not found."
    }
