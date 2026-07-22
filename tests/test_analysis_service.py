"""Focused tests for analysis-run lifecycle transitions."""

from collections.abc import Iterator
from datetime import UTC
from unittest.mock import Mock

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    EvidenceItem,
    EvidenceType,
    Incident,
    IncidentStatus,
)
from app.services.analysis_service import (
    AnalysisAlreadyRunningError,
    AnalysisEvidenceRequiredError,
    AnalysisPersistenceError,
    AnalysisRunNotFoundError,
    AnalysisRunTransitionError,
    AnalysisService,
)
from app.services.incident_service import IncidentNotFoundError


@pytest.fixture
def service_session(
    database_session_factory: sessionmaker[Session],
) -> Iterator[Session]:
    with database_session_factory() as session:
        yield session


def _persist_incident(
    session: Session,
    *,
    with_evidence: bool = True,
) -> Incident:
    incident = Incident(
        public_id="INC-000001",
        name="Checkout failures",
        description="Intermittent checkout errors",
        affected_service="checkout",
        status=IncidentStatus.READY if with_evidence else IncidentStatus.DRAFT,
    )
    if with_evidence:
        incident.evidence_items.append(
            EvidenceItem(
                evidence_code="E-001",
                source_name="checkout.log",
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text="Checkout failed",
                checksum="a" * 64,
            )
        )
    session.add(incident)
    session.commit()
    return incident


def _start_run(service: AnalysisService, public_id: str) -> AnalysisRun:
    return service.start_analysis_run(
        public_id,
        provider_name="fake",
        model_name="fixture-v1",
    )


def test_start_analysis_run_persists_running_lifecycle(
    service_session: Session,
) -> None:
    incident = _persist_incident(service_session)

    analysis_run = _start_run(AnalysisService(service_session), incident.public_id)

    assert analysis_run.id is not None
    assert analysis_run.status is AnalysisRunStatus.RUNNING
    assert analysis_run.started_at.tzinfo is UTC
    assert analysis_run.completed_at is None
    assert analysis_run.error_message is None
    assert analysis_run.provider_name == "fake"
    assert analysis_run.model_name == "fixture-v1"
    assert analysis_run.incident.status is IncidentStatus.ANALYZING


def test_start_analysis_run_requires_existing_incident(
    service_session: Session,
) -> None:
    service = AnalysisService(service_session)

    with pytest.raises(IncidentNotFoundError, match="INC-999999"):
        _start_run(service, "INC-999999")


def test_start_analysis_run_requires_evidence(service_session: Session) -> None:
    incident = _persist_incident(service_session, with_evidence=False)

    with pytest.raises(AnalysisEvidenceRequiredError, match="requires evidence"):
        _start_run(AnalysisService(service_session), incident.public_id)

    assert incident.status is IncidentStatus.DRAFT
    assert service_session.scalar(select(func.count(AnalysisRun.id))) == 0


def test_start_analysis_run_rejects_second_running_run(
    service_session: Session,
) -> None:
    incident = _persist_incident(service_session)
    service = AnalysisService(service_session)
    first_run = _start_run(service, incident.public_id)

    with pytest.raises(AnalysisAlreadyRunningError, match="already has"):
        _start_run(service, incident.public_id)

    assert service_session.scalar(select(func.count(AnalysisRun.id))) == 1
    assert first_run.status is AnalysisRunStatus.RUNNING


def test_mark_analysis_run_completed_sets_terminal_statuses(
    service_session: Session,
) -> None:
    incident = _persist_incident(service_session)
    service = AnalysisService(service_session)
    analysis_run = _start_run(service, incident.public_id)

    completed_run = service.mark_analysis_run_completed(analysis_run.id)

    assert completed_run.status is AnalysisRunStatus.COMPLETED
    assert completed_run.completed_at is not None
    assert completed_run.completed_at.tzinfo is UTC
    assert completed_run.error_message is None
    assert completed_run.incident.status is IncidentStatus.COMPLETED


def test_mark_analysis_run_failed_retains_run_and_safe_explanation(
    service_session: Session,
) -> None:
    incident = _persist_incident(service_session)
    service = AnalysisService(service_session)
    analysis_run = _start_run(service, incident.public_id)

    failed_run = service.mark_analysis_run_failed(
        analysis_run.id,
        error_message="  Structured analysis failed safely.  ",
    )
    service_session.expire_all()
    retained_run = service_session.scalar(
        select(AnalysisRun).where(AnalysisRun.id == analysis_run.id)
    )

    assert failed_run.status is AnalysisRunStatus.FAILED
    assert retained_run is not None
    assert retained_run.status is AnalysisRunStatus.FAILED
    assert retained_run.completed_at is not None
    assert retained_run.error_message == "Structured analysis failed safely."
    assert retained_run.incident.status is IncidentStatus.FAILED


@pytest.mark.parametrize("terminal_status", list(AnalysisRunStatus)[1:])
def test_terminal_analysis_run_cannot_transition_again(
    service_session: Session,
    terminal_status: AnalysisRunStatus,
) -> None:
    incident = _persist_incident(service_session)
    service = AnalysisService(service_session)
    analysis_run = _start_run(service, incident.public_id)
    if terminal_status is AnalysisRunStatus.COMPLETED:
        service.mark_analysis_run_completed(analysis_run.id)
    else:
        service.mark_analysis_run_failed(
            analysis_run.id,
            error_message="Analysis failed safely.",
        )

    with pytest.raises(AnalysisRunTransitionError, match="cannot transition"):
        service.mark_analysis_run_completed(analysis_run.id)


def test_mark_analysis_run_failed_rejects_empty_explanation(
    service_session: Session,
) -> None:
    incident = _persist_incident(service_session)
    service = AnalysisService(service_session)
    analysis_run = _start_run(service, incident.public_id)

    with pytest.raises(ValueError, match="must not be empty"):
        service.mark_analysis_run_failed(analysis_run.id, error_message="  ")

    assert analysis_run.status is AnalysisRunStatus.RUNNING
    assert incident.status is IncidentStatus.ANALYZING


def test_transition_requires_existing_analysis_run(
    service_session: Session,
) -> None:
    with pytest.raises(AnalysisRunNotFoundError, match="999"):
        AnalysisService(service_session).mark_analysis_run_completed(999)


def test_start_analysis_run_rolls_back_persistence_failure(
    service_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incident = _persist_incident(service_session)
    rollback = Mock(wraps=service_session.rollback)
    monkeypatch.setattr(service_session, "rollback", rollback)
    monkeypatch.setattr(
        service_session,
        "commit",
        Mock(side_effect=SQLAlchemyError("database unavailable")),
    )

    with pytest.raises(AnalysisPersistenceError, match="could not be started"):
        _start_run(AnalysisService(service_session), incident.public_id)

    rollback.assert_called_once_with()
    assert service_session.scalar(select(func.count(AnalysisRun.id))) == 0
    persisted_status = service_session.scalar(
        select(Incident.status).where(Incident.id == incident.id)
    )
    assert persisted_status is IncidentStatus.READY


def test_start_analysis_run_refresh_failure_rolls_back_before_commit(
    service_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incident = _persist_incident(service_session)
    rollback = Mock(wraps=service_session.rollback)
    commit = Mock(wraps=service_session.commit)
    monkeypatch.setattr(service_session, "rollback", rollback)
    monkeypatch.setattr(service_session, "commit", commit)
    monkeypatch.setattr(
        service_session,
        "refresh",
        Mock(side_effect=SQLAlchemyError("refresh unavailable")),
    )

    with pytest.raises(AnalysisPersistenceError, match="could not be started"):
        _start_run(AnalysisService(service_session), incident.public_id)

    rollback.assert_called_once_with()
    commit.assert_not_called()
    assert service_session.scalar(select(func.count(AnalysisRun.id))) == 0
    persisted_status = service_session.scalar(
        select(Incident.status).where(Incident.id == incident.id)
    )
    assert persisted_status is IncidentStatus.READY
