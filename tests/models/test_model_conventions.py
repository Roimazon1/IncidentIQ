"""Tests for shared persistence model conventions."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import String
from sqlalchemy.engine import Engine
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.models import (
    ClaimSupportStatus,
    EvidenceType,
    IncidentStatus,
    TimestampMixin,
    UTCDateTime,
    utc_now,
)


class ConventionBase(DeclarativeBase):
    """Isolated declarative base for convention tests."""


class TimestampRecord(TimestampMixin, ConventionBase):
    """Minimal model used to exercise the shared timestamp mixin."""

    __tablename__ = "timestamp_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))


@pytest.fixture
def convention_session_factory(sqlite_engine: Engine) -> sessionmaker[Session]:
    ConventionBase.metadata.create_all(sqlite_engine)
    return sessionmaker(bind=sqlite_engine)


def test_incident_status_values_are_locked() -> None:
    assert [status.value for status in IncidentStatus] == [
        "DRAFT",
        "READY",
        "ANALYZING",
        "COMPLETED",
        "FAILED",
    ]


def test_evidence_type_values_are_locked() -> None:
    assert [evidence_type.value for evidence_type in EvidenceType] == [
        "APPLICATION_LOG",
        "ERROR_TRACE",
        "MONITORING_ALERT",
        "DEPLOYMENT_NOTE",
        "USER_COMPLAINT",
        "API_RESPONSE",
        "DATABASE_ERROR",
        "OTHER",
    ]


def test_claim_support_status_values_are_locked() -> None:
    assert [status.value for status in ClaimSupportStatus] == [
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "INFERRED",
        "CONTRADICTED",
        "UNSUPPORTED",
    ]


def test_utc_now_returns_an_aware_utc_datetime() -> None:
    timestamp = utc_now()

    assert timestamp.tzinfo is UTC
    assert timestamp.utcoffset().total_seconds() == 0


def test_timestamp_columns_are_non_nullable_and_timezone_aware() -> None:
    created_at = TimestampRecord.__table__.c.created_at
    updated_at = TimestampRecord.__table__.c.updated_at

    assert created_at.nullable is False
    assert updated_at.nullable is False
    assert isinstance(created_at.type, UTCDateTime)
    assert isinstance(updated_at.type, UTCDateTime)
    assert created_at.type.impl.timezone is True
    assert updated_at.type.impl.timezone is True
    assert created_at.default is not None
    assert updated_at.default is not None
    assert updated_at.onupdate is not None


def test_timestamp_mixin_round_trips_and_updates_aware_timestamps(
    convention_session_factory: sessionmaker[Session],
) -> None:
    initial_updated_at = datetime(2000, 1, 1, tzinfo=UTC)

    with convention_session_factory() as session:
        record = TimestampRecord(
            name="initial",
            updated_at=initial_updated_at,
        )
        session.add(record)
        session.flush()
        record_id = record.id
        session.commit()

    with convention_session_factory() as session:
        loaded_record = session.get(TimestampRecord, record_id)
        assert loaded_record is not None
        created_at = loaded_record.created_at

        assert created_at.tzinfo is UTC
        assert loaded_record.updated_at.tzinfo is UTC
        assert loaded_record.updated_at == initial_updated_at

        loaded_record.name = "updated"
        session.commit()

    with convention_session_factory() as session:
        reloaded_record = session.get(TimestampRecord, record_id)
        assert reloaded_record is not None

        assert reloaded_record.created_at.tzinfo is UTC
        assert reloaded_record.updated_at.tzinfo is UTC
        assert reloaded_record.created_at == created_at
        assert reloaded_record.updated_at > initial_updated_at


def test_utc_datetime_normalizes_aware_values_to_utc(
    convention_session_factory: sessionmaker[Session],
) -> None:
    source_timestamp = datetime(
        2025,
        1,
        1,
        12,
        tzinfo=timezone(timedelta(hours=2)),
    )
    expected_timestamp = source_timestamp.astimezone(UTC)

    with convention_session_factory() as session:
        record = TimestampRecord(
            name="offset timestamp",
            created_at=source_timestamp,
            updated_at=source_timestamp,
        )
        session.add(record)
        session.flush()
        record_id = record.id
        session.commit()

    with convention_session_factory() as session:
        loaded_record = session.get(TimestampRecord, record_id)
        assert loaded_record is not None

        assert loaded_record.created_at == expected_timestamp
        assert loaded_record.updated_at == expected_timestamp
        assert loaded_record.created_at.tzinfo is UTC
        assert loaded_record.updated_at.tzinfo is UTC


def test_utc_datetime_rejects_naive_values(
    convention_session_factory: sessionmaker[Session],
) -> None:
    with convention_session_factory() as session:
        record = TimestampRecord(
            name="naive timestamp",
            created_at=datetime(2025, 1, 1),
            updated_at=utc_now(),
        )
        session.add(record)

        with pytest.raises(StatementError, match="timezone-aware"):
            session.commit()
