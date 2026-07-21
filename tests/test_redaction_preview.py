"""Focused route tests for the outbound-safe redaction preview."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import EvidenceItem, EvidenceType
from app.schemas.evidence import EvidenceCreate
from app.services.ingestion_service import IngestionService
from tests.evidence_test_support import create_incident, create_incident_through_api


def test_redaction_preview_masks_source_and_content_without_persisting_changes(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    source_secret = "oncall@example.com"
    content_secret = "sk-production-secret-1234"
    original_text = f"api_key={content_secret}&mode=safe"
    with database_session_factory() as session:
        public_id = create_incident(session)
        IngestionService(session).ingest_pasted_text(
            public_id,
            EvidenceCreate(
                source_name=source_secret,
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text=original_text,
            ),
        )

    response = database_client.get(
        f"/incidents/{public_id}/evidence/E-001/redaction-preview"
    )

    assert response.status_code == 200
    assert "Redaction preview" in response.text
    assert source_secret not in response.text
    assert content_secret not in response.text
    assert "[REDACTED_EMAIL]" in response.text
    assert "[REDACTED_API_KEY]" in response.text
    assert "mode=safe" in response.text
    assert "L0001:" in response.text
    assert "2 sensitive values were masked" in response.text
    assert "original evidence remains local and unchanged" in response.text
    assert "original normalized evidence" in response.text
    assert "displayed redacted text" in response.text

    with database_session_factory() as session:
        evidence = session.scalar(select(EvidenceItem))
        assert evidence is not None
        assert evidence.source_name == source_secret
        assert evidence.original_text == original_text
        assert evidence.redacted_text is None


def test_redaction_preview_escapes_content_and_explains_empty_findings(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    original_text = "<script>alert('review')</script>"
    with database_session_factory() as session:
        public_id = create_incident(session)
        IngestionService(session).ingest_pasted_text(
            public_id,
            EvidenceCreate(
                source_name="Operator notes",
                evidence_type=EvidenceType.OTHER,
                original_text=original_text,
            ),
        )

    response = database_client.get(
        f"/incidents/{public_id}/evidence/E-001/redaction-preview"
    )

    assert response.status_code == 200
    assert original_text not in response.text
    assert "&lt;script&gt;alert" in response.text
    assert "No supported sensitive values were detected" in response.text
    assert "Manual review is still required" in response.text


def test_original_evidence_page_links_to_redaction_preview(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as session:
        public_id = create_incident(session)
        IngestionService(session).ingest_pasted_text(
            public_id,
            EvidenceCreate(
                source_name="Checkout log",
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text="checkout failed",
            ),
        )

    response = database_client.get(f"/incidents/{public_id}/evidence/E-001")

    assert response.status_code == 200
    assert "Redaction preview" in response.text
    assert f"/incidents/{public_id}/evidence/E-001/redaction-preview" in response.text


def test_missing_redaction_preview_returns_not_found(
    database_client: TestClient,
) -> None:
    create_incident_through_api(database_client)

    response = database_client.get(
        "/incidents/INC-000001/evidence/E-999/redaction-preview"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Evidence E-999 was not found for incident INC-000001."
    }


@pytest.mark.parametrize(
    ("source_name", "original_text", "parser_detail"),
    [
        (
            "private-json@example.com.json",
            '{"status":',
            "Expecting value",
        ),
        (
            "private-csv@example.com.csv",
            'source,message\napp,"unterminated',
            "unexpected end of data",
        ),
    ],
)
def test_malformed_structured_evidence_returns_safe_validation_page(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
    source_name: str,
    original_text: str,
    parser_detail: str,
) -> None:
    with database_session_factory() as session:
        public_id = create_incident(session)
        IngestionService(session).ingest_pasted_text(
            public_id,
            EvidenceCreate(
                source_name=source_name,
                evidence_type=EvidenceType.API_RESPONSE,
                original_text=original_text,
            ),
        )

    response = database_client.get(
        f"/incidents/{public_id}/evidence/E-001/redaction-preview"
    )

    assert response.status_code == 422
    assert "Redaction preview unavailable" in response.text
    assert "saved structured evidence is malformed" in response.text
    assert "No source metadata or evidence content is shown" in response.text
    assert source_name not in response.text
    assert original_text not in response.text
    assert parser_detail not in response.text
    assert "contains invalid JSON" not in response.text
    assert "contains invalid CSV" not in response.text
