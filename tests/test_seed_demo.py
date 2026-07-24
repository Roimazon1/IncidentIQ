"""Focused tests for idempotent synthetic demo seeding."""

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.models import EvidenceItem, EvidenceType, Incident, IncidentStatus
from app.services.ingestion_service import IngestionService
from scripts.seed_demo import DemoDatasetError, load_demo_definition, seed_demo


DATASET_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "data" / "demo_checkout_incident"
)


def test_first_seed_creates_incident_and_exact_evidence(
    database_session_factory: sessionmaker[Session],
) -> None:
    definition = load_demo_definition(DATASET_DIRECTORY)

    with database_session_factory() as session:
        result = seed_demo(session, DATASET_DIRECTORY)

    assert result.incident_public_id == "INC-000001"
    assert result.incident_created is True
    assert result.added_evidence_codes == ("E-001", "E-002", "E-003", "E-004")

    with database_session_factory() as session:
        incident = session.scalar(
            select(Incident)
            .options(selectinload(Incident.evidence_items))
            .where(Incident.public_id == result.incident_public_id)
        )
        assert incident is not None
        assert incident.name == definition.name
        assert incident.description == definition.description
        assert incident.affected_service == definition.affected_service
        assert incident.reported_start_time is not None
        assert incident.reported_start_time.isoformat() == "2025-02-18T10:05:00+00:00"
        assert incident.status is IncidentStatus.READY

        evidence_items = sorted(
            incident.evidence_items,
            key=lambda item: item.evidence_code,
        )
        assert [item.evidence_code for item in evidence_items] == [
            "E-001",
            "E-002",
            "E-003",
            "E-004",
        ]
        assert [item.source_name for item in evidence_items] == [
            evidence.source_name for evidence in definition.evidence
        ]
        assert [item.evidence_type for item in evidence_items] == [
            EvidenceType(evidence.evidence_type) for evidence in definition.evidence
        ]
        for evidence in evidence_items:
            expected_text = (DATASET_DIRECTORY / evidence.source_name).read_text(
                encoding="utf-8"
            )
            assert evidence.original_text == expected_text
            assert evidence.checksum == IngestionService.calculate_checksum(
                expected_text
            )


def test_repeated_seed_reuses_incident_and_does_not_duplicate_evidence(
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as session:
        first_result = seed_demo(session, DATASET_DIRECTORY)
        original_rows = session.execute(
            select(
                EvidenceItem.id,
                EvidenceItem.evidence_code,
                EvidenceItem.source_name,
                EvidenceItem.checksum,
            ).order_by(EvidenceItem.id)
        ).all()

    with database_session_factory() as session:
        repeated_result = seed_demo(session, DATASET_DIRECTORY)
        repeated_rows = session.execute(
            select(
                EvidenceItem.id,
                EvidenceItem.evidence_code,
                EvidenceItem.source_name,
                EvidenceItem.checksum,
            ).order_by(EvidenceItem.id)
        ).all()
        incident_count = session.scalar(select(func.count(Incident.id)))
        evidence_count = session.scalar(select(func.count(EvidenceItem.id)))

    assert repeated_result.incident_public_id == first_result.incident_public_id
    assert repeated_result.incident_created is False
    assert repeated_result.added_evidence_codes == ()
    assert repeated_rows == original_rows
    assert incident_count == 1
    assert evidence_count == 4


def test_seed_rolls_back_incident_and_evidence_when_final_commit_fails(
    database_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flushed_counts: dict[str, int] = {}

    with database_session_factory() as session:

        def fail_after_flush() -> None:
            flushed_counts["incidents"] = session.scalar(
                select(func.count(Incident.id))
            )
            flushed_counts["evidence"] = session.scalar(
                select(func.count(EvidenceItem.id))
            )
            raise SQLAlchemyError("synthetic commit failure")

        monkeypatch.setattr(session, "commit", fail_after_flush)

        with pytest.raises(
            DemoDatasetError,
            match="could not be seeded",
        ):
            seed_demo(session, DATASET_DIRECTORY)

    assert flushed_counts == {"incidents": 1, "evidence": 4}
    with database_session_factory() as session:
        assert session.scalar(select(func.count(Incident.id))) == 0
        assert session.scalar(select(func.count(EvidenceItem.id))) == 0
