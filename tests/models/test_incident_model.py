"""Tests for the Incident persistence model."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.models import Incident, IncidentStatus


def test_incident_model_fields_and_constraints() -> None:
    mapper = inspect(Incident)
    columns = Incident.__table__.c

    assert set(mapper.columns.keys()) == {
        "id",
        "public_id",
        "name",
        "description",
        "affected_service",
        "reported_start_time",
        "status",
        "created_at",
        "updated_at",
    }
    assert set(mapper.relationships.keys()) == {
        "analysis_runs",
        "evidence_items",
        "reports",
    }
    assert columns.id.primary_key is True
    assert columns.public_id.nullable is False
    assert columns.public_id.unique is True
    assert columns.public_id.index is True
    assert columns.name.nullable is False
    assert columns.description.nullable is False
    assert columns.affected_service.nullable is False
    assert columns.reported_start_time.nullable is True
    assert columns.status.nullable is False
    assert columns.status.type.enum_class is IncidentStatus


def test_incident_persists_with_defaults_and_utc_timestamps(
    model_session_factory: sessionmaker[Session],
) -> None:
    reported_start_time = datetime(2025, 1, 1, 10, tzinfo=UTC)

    with model_session_factory() as session:
        incident = Incident(
            public_id="INC-000001",
            name="Checkout failures",
            description="Intermittent checkout errors",
            affected_service="checkout",
            reported_start_time=reported_start_time,
        )
        session.add(incident)
        session.flush()
        incident_id = incident.id
        session.commit()

    with model_session_factory() as session:
        loaded_incident = session.get(Incident, incident_id)
        assert loaded_incident is not None

        assert loaded_incident.public_id == "INC-000001"
        assert loaded_incident.name == "Checkout failures"
        assert loaded_incident.description == "Intermittent checkout errors"
        assert loaded_incident.affected_service == "checkout"
        assert loaded_incident.reported_start_time == reported_start_time
        assert loaded_incident.status is IncidentStatus.DRAFT
        assert loaded_incident.created_at.tzinfo is UTC
        assert loaded_incident.updated_at.tzinfo is UTC


def test_incident_update_persists_and_advances_updated_at(
    model_session_factory: sessionmaker[Session],
) -> None:
    initial_updated_at = datetime(2020, 1, 1, tzinfo=UTC)

    with model_session_factory() as session:
        incident = Incident(
            public_id="INC-000001",
            name="Checkout failures",
            description="Initial description",
            affected_service="checkout",
            updated_at=initial_updated_at,
        )
        session.add(incident)
        session.flush()
        incident_id = incident.id
        session.commit()

    with model_session_factory() as session:
        incident = session.get(Incident, incident_id)
        assert incident is not None
        incident.description = "Updated investigation description"
        session.commit()

    with model_session_factory() as session:
        updated_incident = session.get(Incident, incident_id)
        assert updated_incident is not None
        assert updated_incident.description == "Updated investigation description"
        assert updated_incident.updated_at > initial_updated_at
        assert updated_incident.updated_at.tzinfo is UTC


def test_incident_public_id_must_be_unique(
    model_session_factory: sessionmaker[Session],
) -> None:
    with model_session_factory() as session:
        session.add_all(
            [
                Incident(
                    public_id="INC-000001",
                    name="First incident",
                    description="First description",
                    affected_service="checkout",
                ),
                Incident(
                    public_id="INC-000001",
                    name="Second incident",
                    description="Second description",
                    affected_service="payments",
                ),
            ]
        )

        with pytest.raises(IntegrityError):
            session.commit()
