"""Focused tests for incident and evidence Pydantic contracts."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.models import EvidenceType, IncidentStatus
from app.schemas import (
    EvidenceCreate,
    EvidenceRead,
    EvidenceUpdate,
    IncidentCreate,
    IncidentRead,
    IncidentUpdate,
)


def test_incident_create_normalizes_text_and_accepts_aware_start_time() -> None:
    start_time = datetime(2025, 1, 1, 10, tzinfo=UTC)

    incident = IncidentCreate(
        name="  Checkout failures  ",
        description="  Intermittent errors  ",
        affected_service="  checkout  ",
        reported_start_time=start_time,
    )

    assert incident.name == "Checkout failures"
    assert incident.description == "Intermittent errors"
    assert incident.affected_service == "checkout"
    assert incident.reported_start_time == start_time


def test_incident_create_rejects_naive_time() -> None:
    with pytest.raises(ValidationError):
        IncidentCreate(
            name="Checkout failures",
            description="Intermittent errors",
            affected_service="checkout",
            reported_start_time=datetime(2025, 1, 1, 10),
        )


def test_incident_create_rejects_generated_fields() -> None:
    with pytest.raises(ValidationError):
        IncidentCreate(
            name="Checkout failures",
            description="Intermittent errors",
            affected_service="checkout",
            public_id="INC-000001",
        )


def test_incident_update_is_partial_and_rejects_blank_values() -> None:
    update = IncidentUpdate(affected_service="  payments  ")

    assert update.model_dump(exclude_unset=True) == {"affected_service": "payments"}
    with pytest.raises(ValidationError):
        IncidentUpdate(name="   ")


def test_empty_incident_update_is_valid_and_excludes_omitted_fields() -> None:
    update = IncidentUpdate()

    assert update.model_dump(exclude_unset=True) == {}


@pytest.mark.parametrize("field_name", ["name", "description", "affected_service"])
def test_incident_update_rejects_explicit_null_for_required_columns(
    field_name: str,
) -> None:
    with pytest.raises(ValidationError):
        IncidentUpdate.model_validate({field_name: None})


def test_incident_update_accepts_explicit_null_to_clear_start_time() -> None:
    update = IncidentUpdate(reported_start_time=None)

    assert update.model_dump(exclude_unset=True) == {"reported_start_time": None}


def test_incident_read_validates_from_attributes() -> None:
    timestamp = datetime(2025, 1, 1, 10, tzinfo=UTC)
    source = SimpleNamespace(
        id=1,
        public_id="INC-000001",
        name="Checkout failures",
        description="Intermittent errors",
        affected_service="checkout",
        reported_start_time=timestamp,
        status=IncidentStatus.DRAFT,
        created_at=timestamp,
        updated_at=timestamp,
    )

    incident = IncidentRead.model_validate(source)

    assert incident.public_id == "INC-000001"
    assert incident.status is IncidentStatus.DRAFT


def test_evidence_create_preserves_original_text_and_parses_enum() -> None:
    original_text = "  2025-01-01 ERROR checkout failed\n"

    evidence = EvidenceCreate(
        source_name="  checkout.log  ",
        evidence_type="APPLICATION_LOG",
        original_text=original_text,
    )

    assert evidence.source_name == "checkout.log"
    assert evidence.evidence_type is EvidenceType.APPLICATION_LOG
    assert evidence.original_text == original_text


def test_evidence_create_rejects_blank_content() -> None:
    with pytest.raises(ValidationError):
        EvidenceCreate(
            source_name="checkout.log",
            evidence_type=EvidenceType.APPLICATION_LOG,
            original_text=" \n\t ",
        )


def test_evidence_create_rejects_generated_fields() -> None:
    with pytest.raises(ValidationError):
        EvidenceCreate(
            source_name="checkout.log",
            evidence_type=EvidenceType.APPLICATION_LOG,
            original_text="ERROR checkout failed",
            evidence_code="E-001",
        )


def test_evidence_update_only_accepts_correctable_metadata() -> None:
    update = EvidenceUpdate(evidence_type="DATABASE_ERROR")

    assert update.model_dump(exclude_unset=True) == {
        "evidence_type": EvidenceType.DATABASE_ERROR
    }
    with pytest.raises(ValidationError):
        EvidenceUpdate(original_text="replacement evidence")


def test_empty_evidence_update_is_valid_and_excludes_omitted_fields() -> None:
    update = EvidenceUpdate()

    assert update.model_dump(exclude_unset=True) == {}


@pytest.mark.parametrize("field_name", ["source_name", "evidence_type"])
def test_evidence_update_rejects_explicit_null_for_required_columns(
    field_name: str,
) -> None:
    with pytest.raises(ValidationError):
        EvidenceUpdate.model_validate({field_name: None})


def test_evidence_read_validates_traceability_fields_from_attributes() -> None:
    timestamp = datetime(2025, 1, 1, 10, tzinfo=UTC)
    source = SimpleNamespace(
        id=2,
        incident_id=1,
        evidence_code="E-001",
        source_name="checkout.log",
        evidence_type=EvidenceType.APPLICATION_LOG,
        original_text="ERROR checkout failed",
        redacted_text="ERROR checkout failed",
        checksum="a" * 64,
        detected_start_time=timestamp,
        detected_end_time=timestamp,
        created_at=timestamp,
        updated_at=timestamp,
    )

    evidence = EvidenceRead.model_validate(source)

    assert evidence.incident_id == 1
    assert evidence.evidence_code == "E-001"
    assert evidence.checksum == "a" * 64
