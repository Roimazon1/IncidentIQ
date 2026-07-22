"""Focused tests for analysis-run lifecycle transitions."""

from collections.abc import Callable, Iterator
from datetime import UTC
from pathlib import Path
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
    Fact,
    Hypothesis,
    Incident,
    IncidentStatus,
    TimelineEvent,
)
from app.schemas.ai_outputs import (
    AIOutput,
    HypothesesOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (
    AIRequest,
    AIResult,
    AnalysisStage,
    OutputSchemaIdentifier,
    PromptName,
    PromptReference,
    PromptVersion,
)
from app.services.analysis_service import (
    AnalysisAlreadyRunningError,
    AnalysisEvidenceRequiredError,
    AnalysisPersistenceError,
    AnalysisProviderRequiredError,
    AnalysisRunNotFoundError,
    AnalysisRunTransitionError,
    AnalysisService,
    AnalysisStageOutputError,
)
from app.services.ai_provider import build_ai_result
from app.services.incident_service import IncidentNotFoundError
from app.services.prompt_registry import PromptRegistry
from app.services.providers.fake_provider import FakeAIProvider


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_ai_responses.json"


class _RecordingProvider:
    def __init__(
        self,
        generate: Callable[[AIRequest], AIResult[AIOutput]],
    ) -> None:
        self._generate = generate
        self.requests: list[AIRequest] = []

    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        self.requests.append(request)
        return self._generate(request)


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
    original_text: str = "Checkout failed",
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
                original_text=original_text,
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


def _recording_stage_provider(fixture_name: str) -> _RecordingProvider:
    registry = PromptRegistry()
    provider = FakeAIProvider.from_file(
        FIXTURE_PATH,
        fixture_name,
        prompt_resolver=registry.resolve_content,
        prompt_bundle_validator=registry.validate_bundle,
    )
    return _RecordingProvider(provider.generate)


def _recording_summary_provider() -> _RecordingProvider:
    return _recording_stage_provider("valid_summary")


def _assert_no_stage_persistence(session: Session, run_id: int) -> None:
    session.expire_all()
    persisted_run = session.scalar(select(AnalysisRun).where(AnalysisRun.id == run_id))
    assert persisted_run is not None
    assert persisted_run.status is AnalysisRunStatus.RUNNING
    assert persisted_run.raw_response is None
    assert persisted_run.prompt_versions == {}
    assert persisted_run.input_evidence_codes == []
    assert session.scalar(select(func.count(Fact.id))) == 0
    assert session.scalar(select(func.count(TimelineEvent.id))) == 0
    assert session.scalar(select(func.count(Hypothesis.id))) == 0


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


def test_summary_stage_extracts_typed_facts_and_assumptions_from_redacted_input(
    service_session: Session,
) -> None:
    secret = "sk-production-secret-1234"
    incident = _persist_incident(
        service_session,
        original_text=f"api_key={secret}\nCheckout failed",
    )
    provider = _recording_summary_provider()
    service = AnalysisService(service_session, ai_provider=provider)
    analysis_run = _start_run(service, incident.public_id)

    result = service.run_summary_stage(analysis_run.id)

    assert isinstance(result.output, SummaryOutputV1)
    assert result.output.summary.text == "Checkout requests are failing."
    assert [fact.claim for fact in result.output.facts] == [
        "The redacted checkout log contains a failure."
    ]
    assert [assumption.claim for assumption in result.output.assumptions] == [
        "A deployment may be related."
    ]
    assert result.output.unknowns == ("The root cause is not verified.",)
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.metadata.analysis_stage is AnalysisStage.SUMMARY
    assert request.output_schema is OutputSchemaIdentifier.SUMMARY_V1
    assert request.prompts.system.name is PromptName.SYSTEM
    assert request.prompts.task.name is PromptName.SUMMARY
    assert request.metadata.incident_public_identifier == incident.public_id
    assert request.metadata.evidence_manifest_checksum is not None
    serialized_request = request.model_dump_json()
    assert secret not in serialized_request
    assert "[REDACTED_API_KEY]" in serialized_request
    assert result.audit.raw_response


def test_summary_stage_rejects_non_summary_output_without_persistence(
    service_session: Session,
) -> None:
    raw_response = '{"events":[],"sensitive":"provider-only"}'

    def generate_timeline_result(request: AIRequest) -> AIResult[AIOutput]:
        return build_ai_result(
            request=request,
            output=TimelineOutputV1(events=()),
            provider_name="fake",
            model_name="fixture-v1",
            attempt_count=1,
            raw_response=raw_response,
        )

    incident = _persist_incident(service_session)
    provider = _RecordingProvider(generate_timeline_result)
    service = AnalysisService(service_session, ai_provider=provider)
    analysis_run = _start_run(service, incident.public_id)

    with pytest.raises(AnalysisStageOutputError) as error_info:
        service.run_summary_stage(analysis_run.id)

    assert raw_response not in str(error_info.value)
    assert raw_response not in repr(error_info.value)
    _assert_no_stage_persistence(service_session, analysis_run.id)


@pytest.mark.parametrize(
    "metadata_update",
    [
        {"analysis_stage": AnalysisStage.TIMELINE},
        {"output_schema": OutputSchemaIdentifier.TIMELINE_V1},
        {
            "system_prompt": PromptReference(
                name=PromptName.SUMMARY,
                version=PromptVersion.V1,
            )
        },
        {
            "task_prompt": PromptReference(
                name=PromptName.TIMELINE,
                version=PromptVersion.V1,
            )
        },
        {"request_identifier": "different-summary-request"},
        {"provider_name": "gemini"},
        {"model_name": "gemini-2.5-flash"},
    ],
    ids=[
        "analysis-stage",
        "output-schema",
        "system-prompt",
        "task-prompt",
        "request-identifier",
        "provider-name",
        "model-name",
    ],
)
def test_summary_stage_rejects_mismatched_traceability_without_persistence(
    service_session: Session,
    metadata_update: dict[str, object],
) -> None:
    fixture_provider = _recording_summary_provider()
    returned_raw_responses: list[str] = []

    def generate_mismatched_result(request: AIRequest) -> AIResult[AIOutput]:
        result = fixture_provider.generate(request)
        returned_raw_responses.append(result.audit.raw_response)
        return AIResult[AIOutput](
            output=result.output,
            metadata=result.metadata.model_copy(update=metadata_update),
            audit=result.audit,
        )

    incident = _persist_incident(service_session)
    provider = _RecordingProvider(generate_mismatched_result)
    service = AnalysisService(service_session, ai_provider=provider)
    analysis_run = _start_run(service, incident.public_id)

    with pytest.raises(AnalysisStageOutputError) as error_info:
        service.run_summary_stage(analysis_run.id)

    assert len(returned_raw_responses) == 1
    raw_response = returned_raw_responses[0]
    assert raw_response not in str(error_info.value)
    assert raw_response not in repr(error_info.value)
    _assert_no_stage_persistence(service_session, analysis_run.id)


def test_summary_stage_requires_injected_provider(
    service_session: Session,
) -> None:
    incident = _persist_incident(service_session)
    service = AnalysisService(service_session)
    analysis_run = _start_run(service, incident.public_id)

    with pytest.raises(AnalysisProviderRequiredError, match="provider is required"):
        service.run_summary_stage(analysis_run.id)

    assert analysis_run.status is AnalysisRunStatus.RUNNING
    assert incident.status is IncidentStatus.ANALYZING


def test_summary_stage_rejects_terminal_run_before_provider_call(
    service_session: Session,
) -> None:
    incident = _persist_incident(service_session)
    provider = _recording_summary_provider()
    service = AnalysisService(service_session, ai_provider=provider)
    analysis_run = _start_run(service, incident.public_id)
    service.mark_analysis_run_completed(analysis_run.id)

    with pytest.raises(AnalysisRunTransitionError, match="summary extraction"):
        service.run_summary_stage(analysis_run.id)

    assert provider.requests == []


def test_timeline_stage_returns_direct_and_inferred_events_from_redacted_input(
    service_session: Session,
) -> None:
    secret = "sk-production-secret-1234"
    incident = _persist_incident(
        service_session,
        original_text=f"api_key={secret}\nCheckout failed",
    )
    provider = _recording_stage_provider("valid_timeline")
    service = AnalysisService(service_session, ai_provider=provider)
    analysis_run = _start_run(service, incident.public_id)

    result = service.run_timeline_stage(analysis_run.id)

    assert isinstance(result.output, TimelineOutputV1)
    direct_event, inferred_event = result.output.events
    assert direct_event.is_inferred is False
    assert direct_event.timestamp == "time unknown"
    assert inferred_event.is_inferred is True
    assert inferred_event.confidence == 70
    assert inferred_event.uncertainty_explanation is not None
    request = provider.requests[0]
    assert request.metadata.analysis_stage is AnalysisStage.TIMELINE
    assert request.output_schema is OutputSchemaIdentifier.TIMELINE_V1
    assert request.prompts.task.name is PromptName.TIMELINE
    assert request.metadata.request_identifier.endswith("-timeline")
    serialized_request = request.model_dump_json()
    assert secret not in serialized_request
    assert "[REDACTED_API_KEY]" in serialized_request
    _assert_no_stage_persistence(service_session, analysis_run.id)


def test_hypotheses_stage_returns_three_ranked_distinct_hypotheses(
    service_session: Session,
) -> None:
    secret = "sk-production-secret-1234"
    incident = _persist_incident(
        service_session,
        original_text=f"api_key={secret}\nCheckout failed",
    )
    provider = _recording_stage_provider("valid_hypotheses")
    service = AnalysisService(service_session, ai_provider=provider)
    analysis_run = _start_run(service, incident.public_id)

    result = service.run_hypotheses_stage(analysis_run.id)

    assert isinstance(result.output, HypothesesOutputV1)
    assert [hypothesis.rank for hypothesis in result.output.hypotheses] == [1, 2, 3]
    assert len({hypothesis.title for hypothesis in result.output.hypotheses}) == 3
    for hypothesis in result.output.hypotheses:
        assert hypothesis.supporting_evidence
        assert hypothesis.missing_evidence
        assert hypothesis.validation_test.description
        assert hypothesis.risk_of_acting
    request = provider.requests[0]
    assert request.metadata.analysis_stage is AnalysisStage.HYPOTHESES
    assert request.output_schema is OutputSchemaIdentifier.HYPOTHESES_V1
    assert request.prompts.task.name is PromptName.HYPOTHESES
    assert request.metadata.request_identifier.endswith("-hypotheses")
    serialized_request = request.model_dump_json()
    assert secret not in serialized_request
    assert "[REDACTED_API_KEY]" in serialized_request
    _assert_no_stage_persistence(service_session, analysis_run.id)


def test_hypotheses_stage_rejects_materially_duplicate_titles_without_persistence(
    service_session: Session,
) -> None:
    fixture_provider = _recording_stage_provider("valid_hypotheses")
    duplicate_titles = ("Same cause", " same   CAUSE ", "SAME CAUSE")
    returned_raw_responses: list[str] = []

    def generate_duplicate_hypotheses(request: AIRequest) -> AIResult[AIOutput]:
        result = fixture_provider.generate(request)
        returned_raw_responses.append(result.audit.raw_response)
        output = result.output
        assert isinstance(output, HypothesesOutputV1)
        duplicate_output = HypothesesOutputV1(
            hypotheses=tuple(
                hypothesis.model_copy(update={"title": duplicate_title})
                for hypothesis, duplicate_title in zip(
                    output.hypotheses,
                    duplicate_titles,
                    strict=True,
                )
            )
        )
        return AIResult[AIOutput](
            output=duplicate_output,
            metadata=result.metadata,
            audit=result.audit,
        )

    incident = _persist_incident(service_session)
    provider = _RecordingProvider(generate_duplicate_hypotheses)
    service = AnalysisService(service_session, ai_provider=provider)
    analysis_run = _start_run(service, incident.public_id)

    with pytest.raises(AnalysisStageOutputError) as error_info:
        service.run_hypotheses_stage(analysis_run.id)

    assert len(returned_raw_responses) == 1
    raw_response = returned_raw_responses[0]
    assert raw_response not in str(error_info.value)
    assert raw_response not in repr(error_info.value)
    _assert_no_stage_persistence(service_session, analysis_run.id)
