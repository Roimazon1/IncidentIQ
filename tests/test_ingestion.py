"""Focused tests for pasted-text and uploaded evidence creation."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import EvidenceItem, EvidenceType, Incident, IncidentStatus
from app.models.evidence import SOURCE_NAME_MAX_LENGTH
from app.routers import evidence as evidence_router
from app.schemas.evidence import EvidenceCreate
from app.schemas.incident import IncidentCreate
from app.services.incident_service import IncidentService
from app.services.ingestion_service import (
    SUPPORTED_UPLOAD_EXTENSIONS,
    EvidenceUpload,
    EvidenceUploadValidationError,
    IngestionService,
)


def _create_incident(session: Session, *, name: str = "Checkout failures") -> str:
    incident = IncidentService(session).create_incident(
        IncidentCreate(
            name=name,
            description="Intermittent checkout errors",
            affected_service="checkout",
        )
    )
    return incident.public_id


def _create_incident_through_api(database_client: TestClient) -> None:
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


def _assert_no_evidence_and_draft(
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


def test_evidence_form_and_pasted_text_post_redirect_flow(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    _create_incident_through_api(database_client)

    form_response = database_client.get(
        "/incidents/INC-000001/evidence/new"
    )
    assert form_response.status_code == 200
    assert "Add evidence" in form_response.text
    assert "Evidence text" in form_response.text
    assert 'type="file"' in form_response.text
    assert f'accept="{",".join(SUPPORTED_UPLOAD_EXTENSIONS)}"' in form_response.text
    readable_formats = (
        f"{', '.join(SUPPORTED_UPLOAD_EXTENSIONS[:-1])}, and "
        f"{SUPPORTED_UPLOAD_EXTENSIONS[-1]}"
    )
    assert f"Supported formats: {readable_formats}." in form_response.text
    assert "multiple" in form_response.text
    assert "Evidence type" in form_response.text
    for evidence_type in EvidenceType:
        assert f'value="{evidence_type.value}"' in form_response.text

    create_response = database_client.post(
        "/incidents/INC-000001/evidence/text",
        data={
            "source_name": "Pasted checkout log",
            "original_text": "ERROR checkout failed",
            "evidence_type": EvidenceType.APPLICATION_LOG.value,
        },
        follow_redirects=False,
    )
    assert create_response.status_code == 303
    assert create_response.headers["location"] == "/incidents/INC-000001"

    with database_session_factory() as session:
        saved_evidence = session.scalar(select(EvidenceItem))
        assert saved_evidence is not None
        assert saved_evidence.evidence_code == "E-001"
        assert saved_evidence.evidence_type is EvidenceType.APPLICATION_LOG
        assert saved_evidence.incident.status is IncidentStatus.READY


def test_all_locked_upload_extensions_persist_in_one_batch(
    database_session_factory: sessionmaker[Session],
) -> None:
    uploads = [
        EvidenceUpload(
            filename=f"evidence{extension}",
            content=f"content for {extension}".encode(),
        )
        for extension in SUPPORTED_UPLOAD_EXTENSIONS
    ]

    with database_session_factory() as session:
        public_id = _create_incident(session)
        saved_uploads = IngestionService(session).ingest_uploaded_files(
            public_id,
            uploads,
        )

    assert [item.evidence_code for item in saved_uploads] == [
        "E-001",
        "E-002",
        "E-003",
        "E-004",
        "E-005",
    ]
    assert [item.source_name for item in saved_uploads] == [
        f"evidence{extension}" for extension in SUPPORTED_UPLOAD_EXTENSIONS
    ]
    assert [item.original_text for item in saved_uploads] == [
        f"content for {extension}" for extension in SUPPORTED_UPLOAD_EXTENSIONS
    ]
    assert all(item.evidence_type is EvidenceType.OTHER for item in saved_uploads)
    assert saved_uploads[0].incident.status is IncidentStatus.READY


def test_multifile_upload_redirects_and_persists_each_file(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    _create_incident_through_api(database_client)

    response = database_client.post(
        "/incidents/INC-000001/evidence/upload",
        data={"evidence_type": EvidenceType.MONITORING_ALERT.value},
        files=[
            ("files", ("checkout.log", b"ERROR checkout failed", "text/plain")),
            (
                "files",
                ("monitoring.csv", b"time,error_rate\n12:00,18", "text/csv"),
            ),
        ],
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/incidents/INC-000001"
    with database_session_factory() as session:
        saved_uploads = list(
            session.scalars(
                select(EvidenceItem).order_by(EvidenceItem.evidence_code)
            )
        )
    assert [item.evidence_code for item in saved_uploads] == ["E-001", "E-002"]
    assert [item.source_name for item in saved_uploads] == [
        "checkout.log",
        "monitoring.csv",
    ]
    assert [item.original_text for item in saved_uploads] == [
        "ERROR checkout failed",
        "time,error_rate\n12:00,18",
    ]
    assert all(
        item.evidence_type is EvidenceType.MONITORING_ALERT
        for item in saved_uploads
    )


def test_saved_evidence_type_can_be_corrected(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    _create_incident_through_api(database_client)
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
    assert response.headers["location"] == "/incidents/INC-000001/evidence/new"
    with database_session_factory() as session:
        saved_evidence = session.scalar(select(EvidenceItem))
        assert saved_evidence is not None
        assert saved_evidence.evidence_type is EvidenceType.DATABASE_ERROR

    form_response = database_client.get("/incidents/INC-000001/evidence/new")
    assert form_response.status_code == 200
    assert "Correct saved classifications" in form_response.text
    assert "E-001" in form_response.text
    assert "Database warning" in form_response.text


def test_type_correction_is_isolated_by_incident_public_id(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as session:
        first_public_id = _create_incident(session)
        second_public_id = _create_incident(session, name="Payment failures")
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
            session.scalars(
                select(EvidenceItem).order_by(EvidenceItem.incident_id)
            )
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
        public_id = _create_incident(session)
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
        public_id = _create_incident(session)
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
    with database_session_factory() as session:
        saved_evidence = session.scalar(select(EvidenceItem))
        assert saved_evidence is not None
        assert saved_evidence.evidence_type is EvidenceType.DEPLOYMENT_NOTE


def test_correcting_missing_evidence_type_returns_not_found(
    database_client: TestClient,
) -> None:
    _create_incident_through_api(database_client)

    response = database_client.post(
        "/incidents/INC-000001/evidence/E-999/type",
        data={"evidence_type": EvidenceType.OTHER.value},
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Evidence E-999 was not found for incident INC-000001."
    }


def test_uploaded_evidence_continues_sequence_after_pasted_evidence(
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as session:
        public_id = _create_incident(session)
        service = IngestionService(session)
        pasted = service.ingest_pasted_text(
            public_id,
            EvidenceCreate(
                source_name="Operator notes",
                evidence_type=EvidenceType.OTHER,
                original_text="Checkout failures began after deployment.",
            ),
        )
        uploaded = service.ingest_uploaded_file(
            public_id,
            EvidenceUpload(
                filename="checkout.log",
                content=b"ERROR checkout failed",
            ),
        )

    assert pasted.evidence_code == "E-001"
    assert uploaded.evidence_code == "E-002"


def test_uploaded_file_checksum_uses_decoded_exact_content(
    database_session_factory: sessionmaker[Session],
) -> None:
    exact_content = "  café checkout failed\r\nnext line\n"

    with database_session_factory() as session:
        public_id = _create_incident(session)
        uploaded = IngestionService(session).ingest_uploaded_file(
            public_id,
            EvidenceUpload(
                filename="checkout.log",
                content=exact_content.encode("utf-8"),
            ),
        )

    assert uploaded.original_text == exact_content
    assert uploaded.checksum == IngestionService.calculate_checksum(exact_content)


def test_uploaded_filename_is_sanitized_before_persistence(
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as session:
        public_id = _create_incident(session)
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
        public_id = _create_incident(session)
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
    _create_incident_through_api(database_client)

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
    _assert_no_evidence_and_draft(database_session_factory)


def test_oversized_upload_returns_clear_error_and_creates_nothing(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evidence_router.settings, "max_upload_bytes", 4)
    _create_incident_through_api(database_client)

    response = database_client.post(
        "/incidents/INC-000001/evidence/upload",
        files=[("files", ("checkout.log", b"12345", "text/plain"))],
    )

    assert response.status_code == 422
    assert "checkout.log exceeds the maximum upload size of 4 bytes" in response.text
    _assert_no_evidence_and_draft(database_session_factory)


def test_invalid_utf8_upload_returns_clear_error_and_creates_nothing(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    _create_incident_through_api(database_client)

    response = database_client.post(
        "/incidents/INC-000001/evidence/upload",
        files=[("files", ("checkout.log", b"\xff\xfe\xfd", "text/plain"))],
    )

    assert response.status_code == 422
    assert "checkout.log must contain valid UTF-8 text" in response.text
    _assert_no_evidence_and_draft(database_session_factory)


def test_binary_control_upload_returns_clear_error_and_creates_nothing(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    _create_incident_through_api(database_client)

    response = database_client.post(
        "/incidents/INC-000001/evidence/upload",
        files=[("files", ("checkout.log", b"header\x00value", "text/plain"))],
    )

    assert response.status_code == 422
    assert "checkout.log contains unreadable binary content" in response.text
    _assert_no_evidence_and_draft(database_session_factory)


def test_upload_to_missing_incident_returns_not_found(
    database_client: TestClient,
) -> None:
    response = database_client.post(
        "/incidents/INC-999999/evidence/upload",
        files=[
            ("files", ("checkout.log", b"ERROR checkout failed", "text/plain"))
        ],
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Incident INC-999999 was not found."
    }


def test_blank_pasted_evidence_preserves_form_and_creates_nothing(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    _create_incident_through_api(database_client)
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
    _assert_no_evidence_and_draft(database_session_factory)


def test_get_evidence_form_for_missing_incident_returns_not_found(
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
