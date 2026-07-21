"""Focused tests for saved evidence classification and preview behavior."""

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import EvidenceItem, EvidenceType
from app.schemas.evidence import EvidenceCreate
from app.services.ingestion_service import EvidenceUpload, IngestionService
from tests.evidence_test_support import (
    assert_active_evidence_tab,
    create_incident,
    create_incident_through_api,
)


def test_evidence_preview_renders_metadata_and_escaped_original_content(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    original_text = "<script>unsafe</script>\ncheckout & payment"
    with database_session_factory() as session:
        public_id = create_incident(session)
        evidence = IngestionService(session).ingest_pasted_text(
            public_id,
            EvidenceCreate(
                source_name="Operator <notes>",
                evidence_type=EvidenceType.USER_COMPLAINT,
                original_text=original_text,
            ),
        )
        checksum = evidence.checksum

    response = database_client.get(f"/incidents/{public_id}/evidence/E-001")

    assert response.status_code == 200
    assert "Evidence E-001" in response.text
    assert "Operator &lt;notes&gt;" in response.text
    assert EvidenceType.USER_COMPLAINT.value in response.text
    assert checksum in response.text
    assert (
        "&lt;script&gt;unsafe&lt;/script&gt;\ncheckout &amp; payment" in response.text
    )
    assert original_text not in response.text

    form_response = database_client.get(f"/incidents/{public_id}/evidence/new")
    assert f"/incidents/{public_id}/evidence/E-001" in form_response.text


def test_evidence_preview_is_scoped_to_the_incident(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as session:
        first_public_id = create_incident(session)
        second_public_id = create_incident(session, name="Payment failures")
        service = IngestionService(session)
        service.ingest_pasted_text(
            first_public_id,
            EvidenceCreate(
                source_name="Checkout log",
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text="checkout-only evidence",
            ),
        )
        service.ingest_pasted_text(
            second_public_id,
            EvidenceCreate(
                source_name="Payment log",
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text="payment-only evidence",
            ),
        )

    response = database_client.get(f"/incidents/{first_public_id}/evidence/E-001")

    assert response.status_code == 200
    assert "Checkout log" in response.text
    assert "checkout-only evidence" in response.text
    assert "Payment log" not in response.text
    assert "payment-only evidence" not in response.text


def test_missing_evidence_preview_returns_not_found(
    database_client: TestClient,
) -> None:
    create_incident_through_api(database_client)

    response = database_client.get("/incidents/INC-000001/evidence/E-999")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Evidence E-999 was not found for incident INC-000001."
    }


def test_saved_evidence_type_can_be_corrected(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    create_incident_through_api(database_client)
    database_client.post(
        "/incidents/INC-000001/evidence/text",
        data={
            "source_name": "Database warning",
            "original_text": "Connection pool exhausted",
            "evidence_type": EvidenceType.OTHER.value,
        },
    )

    response = database_client.post(
        "/incidents/INC-000001/evidence/E-001/type",
        data={"evidence_type": EvidenceType.DATABASE_ERROR.value},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == (
        "/incidents/INC-000001/evidence/new?tab=saved"
    )
    with database_session_factory() as session:
        saved_evidence = session.scalar(select(EvidenceItem))
        assert saved_evidence is not None
        assert saved_evidence.evidence_type is EvidenceType.DATABASE_ERROR

    form_response = database_client.get(response.headers["location"])
    assert form_response.status_code == 200
    assert_active_evidence_tab(form_response, "saved")
    assert "Correct saved classifications" in form_response.text
    assert "E-001" in form_response.text
    assert "Database warning" in form_response.text


def test_evidence_form_tab_query_is_validated(
    database_client: TestClient,
) -> None:
    create_incident_through_api(database_client)

    saved_response = database_client.get("/incidents/INC-000001/evidence/new?tab=saved")
    invalid_response = database_client.get(
        "/incidents/INC-000001/evidence/new?tab=unknown"
    )

    assert saved_response.status_code == 200
    assert_active_evidence_tab(saved_response, "saved")
    assert invalid_response.status_code == 422


def test_type_correction_is_isolated_by_incident_public_id(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as session:
        first_public_id = create_incident(session)
        second_public_id = create_incident(session, name="Payment failures")
        service = IngestionService(session)
        service.ingest_pasted_text(
            first_public_id,
            EvidenceCreate(
                source_name="Checkout log",
                evidence_type=EvidenceType.OTHER,
                original_text="Checkout failed",
            ),
        )
        service.ingest_pasted_text(
            second_public_id,
            EvidenceCreate(
                source_name="Payment log",
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text="Payment timed out",
            ),
        )

    response = database_client.post(
        f"/incidents/{first_public_id}/evidence/E-001/type",
        data={"evidence_type": EvidenceType.DATABASE_ERROR.value},
        follow_redirects=False,
    )

    assert response.status_code == 303
    with database_session_factory() as session:
        saved_evidence = list(
            session.scalars(select(EvidenceItem).order_by(EvidenceItem.incident_id))
        )
    assert [item.evidence_code for item in saved_evidence] == ["E-001", "E-001"]
    assert [item.evidence_type for item in saved_evidence] == [
        EvidenceType.DATABASE_ERROR,
        EvidenceType.APPLICATION_LOG,
    ]


def test_type_correction_preserves_original_evidence_data(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    exact_content = "  checkout failed\r\nconnection pool exhausted\n"
    with database_session_factory() as session:
        public_id = create_incident(session)
        original = IngestionService(session).ingest_uploaded_file(
            public_id,
            EvidenceUpload(
                filename="checkout.log",
                content=exact_content.encode("utf-8"),
                evidence_type=EvidenceType.APPLICATION_LOG,
            ),
        )
        original_values = (
            original.id,
            original.evidence_code,
            original.source_name,
            original.original_text,
            original.checksum,
        )

    response = database_client.post(
        f"/incidents/{public_id}/evidence/E-001/type",
        data={"evidence_type": EvidenceType.DATABASE_ERROR.value},
        follow_redirects=False,
    )

    assert response.status_code == 303
    with database_session_factory() as session:
        corrected = session.scalar(select(EvidenceItem))
        assert corrected is not None
        assert corrected.evidence_type is EvidenceType.DATABASE_ERROR
        assert (
            corrected.id,
            corrected.evidence_code,
            corrected.source_name,
            corrected.original_text,
            corrected.checksum,
        ) == original_values


def test_invalid_evidence_type_is_rejected_without_changing_saved_type(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as session:
        public_id = create_incident(session)
        IngestionService(session).ingest_pasted_text(
            public_id,
            EvidenceCreate(
                source_name="Deployment notes",
                evidence_type=EvidenceType.DEPLOYMENT_NOTE,
                original_text="Deployed v2.4.1",
            ),
        )

    response = database_client.post(
        f"/incidents/{public_id}/evidence/E-001/type",
        data={"evidence_type": "NOT_A_REAL_TYPE"},
    )

    assert response.status_code == 422
    assert "Input should be" in response.text
    assert_active_evidence_tab(response, "saved")
    with database_session_factory() as session:
        saved_evidence = session.scalar(select(EvidenceItem))
        assert saved_evidence is not None
        assert saved_evidence.evidence_type is EvidenceType.DEPLOYMENT_NOTE


def test_correcting_missing_evidence_type_returns_not_found(
    database_client: TestClient,
) -> None:
    create_incident_through_api(database_client)

    response = database_client.post(
        "/incidents/INC-000001/evidence/E-999/type",
        data={"evidence_type": EvidenceType.OTHER.value},
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Evidence E-999 was not found for incident INC-000001."
    }
