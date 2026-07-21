"""Tests for the EvidenceItem persistence model and ownership rules."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import UniqueConstraint, delete, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.models import EvidenceItem, EvidenceType, Incident


def test_evidence_item_fields_constraints_and_relationships() -> None:
    mapper = inspect(EvidenceItem)
    columns = EvidenceItem.__table__.c
    unique_column_sets = {
        tuple(column.name for column in constraint.columns)
        for constraint in EvidenceItem.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert set(mapper.columns.keys()) == {
        "id",
        "incident_id",
        "evidence_code",
        "source_name",
        "evidence_type",
        "original_text",
        "redacted_text",
        "checksum",
        "detected_start_time",
        "detected_end_time",
        "created_at",
        "updated_at",
    }
    assert set(mapper.relationships.keys()) == {"incident"}
    assert mapper.relationships.incident.back_populates == "evidence_items"
    assert "delete-orphan" in inspect(Incident).relationships.evidence_items.cascade
    assert columns.id.primary_key is True
    assert columns.incident_id.nullable is False
    assert columns.incident_id.index is True
    incident_foreign_key = next(iter(columns.incident_id.foreign_keys))
    assert incident_foreign_key.target_fullname == "incidents.id"
    assert columns.evidence_code.nullable is False
    assert columns.source_name.nullable is False
    assert columns.evidence_type.nullable is False
    assert columns.evidence_type.type.enum_class is EvidenceType
    assert columns.original_text.nullable is False
    assert columns.redacted_text.nullable is True
    assert columns.checksum.nullable is False
    assert columns.detected_start_time.nullable is True
    assert columns.detected_end_time.nullable is True
    assert ("incident_id", "evidence_code") in unique_column_sets


def test_evidence_item_persists_original_and_redacted_text_separately(
    model_session_factory: sessionmaker[Session],
) -> None:
    detected_start_time = datetime(2025, 1, 1, 10, tzinfo=UTC)

    with model_session_factory() as session:
        incident = Incident(
            public_id="INC-000001",
            name="Checkout failures",
            description="Intermittent checkout errors",
            affected_service="checkout",
        )
        evidence = EvidenceItem(
            evidence_code="E-001",
            source_name="checkout.log",
            evidence_type=EvidenceType.APPLICATION_LOG,
            original_text="token=secret",
            redacted_text="token=[REDACTED]",
            checksum="a" * 64,
            detected_start_time=detected_start_time,
        )
        incident.evidence_items.append(evidence)
        session.add(incident)
        session.flush()
        evidence_id = evidence.id
        session.commit()

    with model_session_factory() as session:
        loaded_evidence = session.get(EvidenceItem, evidence_id)
        assert loaded_evidence is not None

        assert loaded_evidence.incident.public_id == "INC-000001"
        assert loaded_evidence.evidence_code == "E-001"
        assert loaded_evidence.evidence_type is EvidenceType.APPLICATION_LOG
        assert loaded_evidence.original_text == "token=secret"
        assert loaded_evidence.redacted_text == "token=[REDACTED]"
        assert loaded_evidence.checksum == "a" * 64
        assert loaded_evidence.detected_start_time == detected_start_time
        assert loaded_evidence.detected_end_time is None
        assert loaded_evidence.created_at.tzinfo is UTC
        assert loaded_evidence.updated_at.tzinfo is UTC


def test_evidence_code_is_unique_only_within_its_incident(
    model_session_factory: sessionmaker[Session],
) -> None:
    with model_session_factory() as session:
        first_incident = Incident(
            public_id="INC-000001",
            name="First incident",
            description="First description",
            affected_service="checkout",
        )
        second_incident = Incident(
            public_id="INC-000002",
            name="Second incident",
            description="Second description",
            affected_service="payments",
        )
        first_incident.evidence_items.append(
            EvidenceItem(
                evidence_code="E-001",
                source_name="first.log",
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text="first",
                checksum="a" * 64,
            )
        )
        second_incident.evidence_items.append(
            EvidenceItem(
                evidence_code="E-001",
                source_name="second.log",
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text="second",
                checksum="b" * 64,
            )
        )
        session.add_all([first_incident, second_incident])
        session.flush()
        first_incident_id = first_incident.id
        session.commit()

    with model_session_factory() as session:
        session.add(
            EvidenceItem(
                incident_id=first_incident_id,
                evidence_code="E-001",
                source_name="duplicate.log",
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text="duplicate",
                checksum="c" * 64,
            )
        )

        with pytest.raises(IntegrityError):
            session.commit()


def test_deleting_incident_cascades_to_evidence_items(
    model_session_factory: sessionmaker[Session],
) -> None:
    with model_session_factory() as session:
        incident = Incident(
            public_id="INC-000001",
            name="Checkout failures",
            description="Intermittent checkout errors",
            affected_service="checkout",
            evidence_items=[
                EvidenceItem(
                    evidence_code="E-001",
                    source_name="checkout.log",
                    evidence_type=EvidenceType.APPLICATION_LOG,
                    original_text="failure",
                    checksum="a" * 64,
                )
            ],
        )
        session.add(incident)
        session.flush()
        incident_id = incident.id
        evidence_id = incident.evidence_items[0].id
        session.commit()

    with model_session_factory() as session:
        incident = session.get(Incident, incident_id)
        assert incident is not None
        session.delete(incident)
        session.commit()

    with model_session_factory() as session:
        assert session.get(Incident, incident_id) is None
        assert session.get(EvidenceItem, evidence_id) is None


def test_evidence_item_requires_an_existing_incident(
    model_session_factory: sessionmaker[Session],
) -> None:
    with model_session_factory() as session:
        session.add(
            EvidenceItem(
                incident_id=999,
                evidence_code="E-001",
                source_name="orphan.log",
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text="orphan",
                checksum="a" * 64,
            )
        )

        with pytest.raises(IntegrityError):
            session.commit()


def test_database_level_incident_delete_cascades_to_evidence(
    sqlite_engine: Engine,
    model_session_factory: sessionmaker[Session],
) -> None:
    with model_session_factory() as session:
        incident = Incident(
            public_id="INC-000001",
            name="Checkout failures",
            description="Intermittent checkout errors",
            affected_service="checkout",
            evidence_items=[
                EvidenceItem(
                    evidence_code="E-001",
                    source_name="checkout.log",
                    evidence_type=EvidenceType.APPLICATION_LOG,
                    original_text="failure",
                    checksum="a" * 64,
                )
            ],
        )
        session.add(incident)
        session.flush()
        incident_id = incident.id
        evidence_id = incident.evidence_items[0].id
        session.commit()

    with sqlite_engine.begin() as connection:
        result = connection.execute(
            delete(Incident).where(Incident.id == incident_id),
        )
        assert result.rowcount == 1

    with model_session_factory() as session:
        assert session.get(Incident, incident_id) is None
        assert session.get(EvidenceItem, evidence_id) is None
