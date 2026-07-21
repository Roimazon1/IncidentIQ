"""Focused tests for evidence upload validation."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.models.evidence import SOURCE_NAME_MAX_LENGTH
from app.routers import evidence as evidence_router
from app.services.ingestion_service import (
    EvidenceUpload,
    EvidenceUploadValidationError,
    IngestionService,
)
from tests.evidence_test_support import (
    assert_active_evidence_tab,
    assert_no_evidence_and_draft,
    create_incident,
    create_incident_through_api,
)


def test_uploaded_filename_is_sanitized_before_persistence(
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as session:
        public_id = create_incident(session)
        uploaded = IngestionService(session).ingest_uploaded_file(
            public_id,
            EvidenceUpload(
                filename=r"..\..\Checkout log (prod)#1.LOG",
                content=b"ERROR checkout failed",
            ),
        )

    assert uploaded.source_name == "Checkout_log_prod_1.LOG"


def test_filename_sanitization_rejects_a_missing_basename() -> None:
    with pytest.raises(
        EvidenceUploadValidationError,
        match="must include a valid filename",
    ):
        IngestionService.sanitize_filename("../../")


def test_filename_sanitization_rejects_overlong_filename() -> None:
    overlong_filename = f"{'a' * SOURCE_NAME_MAX_LENGTH}.log"

    with pytest.raises(
        EvidenceUploadValidationError,
        match=f"{SOURCE_NAME_MAX_LENGTH} characters or fewer",
    ):
        IngestionService.sanitize_filename(overlong_filename)


def test_upload_at_exact_size_limit_is_accepted(
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as session:
        public_id = create_incident(session)
        uploaded = IngestionService(
            session,
            max_upload_bytes=4,
        ).ingest_uploaded_file(
            public_id,
            EvidenceUpload(filename="note.txt", content=b"1234"),
        )

    assert uploaded.original_text == "1234"


def test_unsupported_extension_rejects_the_entire_upload_batch(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    create_incident_through_api(database_client)

    response = database_client.post(
        "/incidents/INC-000001/evidence/upload",
        files=[
            ("files", ("checkout.log", b"ERROR checkout failed", "text/plain")),
            ("files", ("payload.exe", b"do not execute", "application/octet-stream")),
        ],
    )

    assert response.status_code == 422
    assert "payload.exe has an unsupported extension" in response.text
    assert ".txt, .log, .json, .csv, .md" in response.text
    assert_active_evidence_tab(response, "upload")
    assert_no_evidence_and_draft(database_session_factory)


def test_oversized_upload_returns_clear_error_and_creates_nothing(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evidence_router.settings, "max_upload_bytes", 4)
    create_incident_through_api(database_client)

    response = database_client.post(
        "/incidents/INC-000001/evidence/upload",
        files=[("files", ("checkout.log", b"12345", "text/plain"))],
    )

    assert response.status_code == 422
    assert "checkout.log exceeds the maximum upload size of 4 bytes" in response.text
    assert_no_evidence_and_draft(database_session_factory)


def test_empty_upload_returns_clear_error_and_creates_nothing(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    create_incident_through_api(database_client)

    response = database_client.post(
        "/incidents/INC-000001/evidence/upload",
        files=[("files", ("empty.log", b"", "text/plain"))],
    )

    assert response.status_code == 422
    assert "evidence text must not be blank" in response.text
    assert_no_evidence_and_draft(database_session_factory)


def test_invalid_utf8_upload_returns_clear_error_and_creates_nothing(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    create_incident_through_api(database_client)

    response = database_client.post(
        "/incidents/INC-000001/evidence/upload",
        files=[("files", ("checkout.log", b"\xff\xfe\xfd", "text/plain"))],
    )

    assert response.status_code == 422
    assert "checkout.log must contain valid UTF-8 text" in response.text
    assert_no_evidence_and_draft(database_session_factory)


def test_binary_control_upload_returns_clear_error_and_creates_nothing(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    create_incident_through_api(database_client)

    response = database_client.post(
        "/incidents/INC-000001/evidence/upload",
        files=[("files", ("checkout.log", b"header\x00value", "text/plain"))],
    )

    assert response.status_code == 422
    assert "checkout.log contains unreadable binary content" in response.text
    assert_no_evidence_and_draft(database_session_factory)
