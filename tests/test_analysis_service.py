"""Focused tests for analysis-run lifecycle transitions."""

import json
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.config import Settings
from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    ClaimSupportStatus,
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
    ContradictingEvidenceV1,
    CriticOutputV1,
    EvidenceReferenceV1,
    HypothesesOutputV1,
    OpenQuestionsOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (
    AIFailureCategory,
    AIRequest,
    AIResult,
    AnalysisStage,
    CriticContextV1,
    OutputSchemaIdentifier,
    PromptName,
    PromptReference,
    PromptVersion,
    SuccessAuditData,
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
from app.services.analysis_service_factory import build_configured_analysis_service
from app.services.ai_provider import AIProviderExecutionError, build_ai_result
from app.services.incident_service import IncidentNotFoundError
from app.services.prompt_registry import PromptRegistry
from app.services.providers.fake_provider import FakeAIProvider
from app.services.validation_service import (
    EvidenceReferenceValidationStatus,
    ValidationService,
)


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
    original_text: str = "checkout failed",
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


def _fixture_provider(fixture_name: str) -> FakeAIProvider:
    registry = PromptRegistry()
    return FakeAIProvider.from_file(
        FIXTURE_PATH,
        fixture_name,
        prompt_resolver=registry.resolve_content,
        prompt_bundle_validator=registry.validate_bundle,
    )


def _recording_stage_provider(fixture_name: str) -> _RecordingProvider:
    return _RecordingProvider(_fixture_provider(fixture_name).generate)


def _recording_summary_provider() -> _RecordingProvider:
    return _recording_stage_provider("valid_summary")


def _recording_core_provider(
    *,
    timeline_fixture: str = "valid_timeline",
) -> _RecordingProvider:
    providers = {
        OutputSchemaIdentifier.SUMMARY_V1: _fixture_provider("valid_summary"),
        OutputSchemaIdentifier.TIMELINE_V1: _fixture_provider(timeline_fixture),
        OutputSchemaIdentifier.HYPOTHESES_V1: _fixture_provider("valid_hypotheses"),
        OutputSchemaIdentifier.CRITIC_V1: _fixture_provider("valid_critic"),
        OutputSchemaIdentifier.REASONING_RISKS_V1: _fixture_provider("valid_bias"),
        OutputSchemaIdentifier.OPEN_QUESTIONS_V1: _fixture_provider(
            "valid_open_questions"
        ),
    }

    def generate(request: AIRequest) -> AIResult[AIOutput]:
        return providers[request.output_schema].generate(request)

    return _RecordingProvider(generate)


def _critic_context() -> CriticContextV1:
    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return CriticContextV1(
        summary=SummaryOutputV1.model_validate_json(
            fixture_bank["valid_summary"]["raw_response"]
        ),
        timeline=TimelineOutputV1.model_validate_json(
            fixture_bank["valid_timeline"]["raw_response"]
        ),
        hypotheses=HypothesesOutputV1.model_validate_json(
            fixture_bank["valid_hypotheses"]["raw_response"]
        ),
    )


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


def test_configured_fake_analysis_loads_only_app_owned_runtime_fixtures(
    service_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_text = Path.read_text
    read_paths: list[Path] = []

    def recording_read_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        read_paths.append(path)
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", recording_read_text)

    service = build_configured_analysis_service(
        service_session,
        Settings.model_validate(
            {
                "ai_provider": "fake",
                "gemini_api_key": None,
                "gemini_model": None,
            }
        ),
    )

    runtime_fixture_paths = [
        path for path in read_paths if path.name == "fake_ai_core_responses.json"
    ]
    expected_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "resources"
        / "fake_ai_core_responses.json"
    )
    assert isinstance(service, AnalysisService)
    assert runtime_fixture_paths == [expected_path]
    assert all(part.casefold() != "tests" for part in expected_path.parts)


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


def test_competing_analysis_starts_create_one_running_run_and_one_safe_conflict(
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as setup_session:
        incident = _persist_incident(setup_session)
        public_id = incident.public_id

    start_barrier = Barrier(2)

    def start_competing_run() -> tuple[str, int | str]:
        with database_session_factory() as session:
            start_barrier.wait(timeout=5)
            try:
                analysis_run = _start_run(AnalysisService(session), public_id)
            except AnalysisAlreadyRunningError as exc:
                return "conflict", str(exc)
            return "started", analysis_run.id

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: start_competing_run(), range(2)))

    assert sorted(result[0] for result in results) == ["conflict", "started"]
    conflict = next(result for result in results if result[0] == "conflict")
    assert conflict[1] == (f"Incident {public_id} already has a running analysis.")
    with database_session_factory() as verification_session:
        running_runs = list(
            verification_session.scalars(
                select(AnalysisRun).where(
                    AnalysisRun.incident.has(public_id=public_id),
                    AnalysisRun.status == AnalysisRunStatus.RUNNING,
                )
            )
        )
    assert len(running_runs) == 1


def test_mark_analysis_run_completed_rejects_missing_stage_results(
    service_session: Session,
) -> None:
    incident = _persist_incident(service_session)
    service = AnalysisService(service_session)
    analysis_run = _start_run(service, incident.public_id)

    with pytest.raises(AnalysisRunTransitionError, match="required stage results"):
        service.mark_analysis_run_completed(analysis_run.id)

    assert analysis_run.status is AnalysisRunStatus.RUNNING
    assert analysis_run.completed_at is None
    assert analysis_run.incident.status is IncidentStatus.ANALYZING


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
    if terminal_status is AnalysisRunStatus.FAILED:
        service.mark_analysis_run_failed(
            analysis_run.id,
            error_message="Analysis failed safely.",
        )
    else:
        analysis_run.status = AnalysisRunStatus.COMPLETED
        incident.status = IncidentStatus.COMPLETED
        service_session.commit()

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
        original_text=f"api_key={secret}\ncheckout failed",
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


@pytest.mark.parametrize(
    ("reference_update", "expected_status"),
    [
        (
            {"evidence_id": "E-999"},
            EvidenceReferenceValidationStatus.UNKNOWN_EVIDENCE_ID,
        ),
        (
            {"excerpt": "fabricated database outage"},
            EvidenceReferenceValidationStatus.EXCERPT_MISMATCH,
        ),
        (
            {"line_range": "999"},
            EvidenceReferenceValidationStatus.INVALID_LINE_RANGE,
        ),
    ],
    ids=["unknown-evidence-id", "fabricated-excerpt", "out-of-range-line"],
)
def test_summary_stage_reports_invalid_traceability_without_terminating_run(
    service_session: Session,
    reference_update: dict[str, str],
    expected_status: EvidenceReferenceValidationStatus,
) -> None:
    fixture_provider = _recording_summary_provider()
    raw_audit_marker = "raw-audit-only-sensitive-provider-content"

    def generate_invalid_reference(request: AIRequest) -> AIResult[AIOutput]:
        result = fixture_provider.generate(request)
        output = result.output
        assert isinstance(output, SummaryOutputV1)
        fact = output.facts[0]
        invalid_reference = fact.evidence[0].model_copy(update=reference_update)
        invalid_output = output.model_copy(
            update={
                "facts": (fact.model_copy(update={"evidence": (invalid_reference,)}),)
            }
        )
        return AIResult[AIOutput](
            output=invalid_output,
            metadata=result.metadata,
            audit=SuccessAuditData(raw_response=raw_audit_marker),
        )

    incident = _persist_incident(
        service_session,
        original_text="checkout failed\nretry=false",
    )
    provider = _RecordingProvider(generate_invalid_reference)
    service = AnalysisService(service_session, ai_provider=provider)
    analysis_run = _start_run(service, incident.public_id)

    result = service.run_summary_stage(analysis_run.id)
    outcomes = ValidationService.validate_output_references(
        result.output,
        provider.requests[0].evidence_manifest,
    )

    assert len(outcomes) == 1
    assert outcomes[0].status is expected_status
    assert outcomes[0].is_valid is False
    assert result.audit.raw_response == raw_audit_marker
    assert raw_audit_marker not in result.model_dump_json()
    assert raw_audit_marker not in repr(result)
    assert raw_audit_marker not in repr(outcomes)
    _assert_no_stage_persistence(service_session, analysis_run.id)


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
    service.mark_analysis_run_failed(
        analysis_run.id,
        error_message="Analysis failed safely.",
    )

    with pytest.raises(AnalysisRunTransitionError, match="summary extraction"):
        service.run_summary_stage(analysis_run.id)

    assert provider.requests == []


def test_timeline_stage_returns_direct_and_inferred_events_from_redacted_input(
    service_session: Session,
) -> None:
    secret = "sk-production-secret-1234"
    incident = _persist_incident(
        service_session,
        original_text=(f"api_key={secret}\n2025-01-01T12:30:00+02:00 checkout failed"),
    )
    provider = _recording_stage_provider("valid_timeline")
    service = AnalysisService(service_session, ai_provider=provider)
    analysis_run = _start_run(service, incident.public_id)

    result = service.run_timeline_stage(analysis_run.id)

    assert isinstance(result.output, TimelineOutputV1)
    direct_event, inferred_event = result.output.events
    assert direct_event.is_inferred is False
    assert direct_event.timestamp == "2025-01-01T12:30:00+02:00"
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
        original_text=f"api_key={secret}\ncheckout failed",
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


def test_critic_stage_returns_separate_typed_redacted_result(
    service_session: Session,
) -> None:
    secret = "sk-production-secret-1234"
    incident = _persist_incident(
        service_session,
        original_text=f"api_key={secret}\ncheckout failed",
    )
    provider = _recording_stage_provider("valid_critic")
    service = AnalysisService(service_session, ai_provider=provider)
    analysis_run = _start_run(service, incident.public_id)

    critic_context = _critic_context()
    result = service.run_critic_stage(
        analysis_run.id,
        critic_context=critic_context,
    )

    assert isinstance(result.output, CriticOutputV1)
    assert result.output.findings[0].affected_claim == (
        "Database connection pool exhaustion"
    )
    assert result.output.alternative_hypothesis is not None
    assert result.output.alternative_hypothesis.hypothesis_id == "H-003"
    request = provider.requests[0]
    assert request.metadata.analysis_stage is AnalysisStage.CRITIC
    assert request.output_schema is OutputSchemaIdentifier.CRITIC_V1
    assert request.prompts.task.name is PromptName.CRITIC
    assert request.metadata.request_identifier.endswith("-critic")
    assert request.critic_context == critic_context
    assert request.critic_context.hypotheses.hypotheses[0].title == (
        "Database connection pool exhaustion"
    )
    serialized_request = request.model_dump_json()
    assert secret not in serialized_request
    assert "[REDACTED_API_KEY]" in serialized_request
    assert '"raw_response"' not in serialized_request
    _assert_no_stage_persistence(service_session, analysis_run.id)


def test_core_analysis_persists_all_validated_outputs_and_audit_data(
    database_session_factory: sessionmaker[Session],
) -> None:
    secret = "sk-production-secret-1234"
    provider = _recording_core_provider()
    with database_session_factory() as session:
        incident = _persist_incident(
            session,
            original_text=(
                f"api_key={secret}\n2025-01-01T12:30:00+02:00 checkout failed"
            ),
        )
        service = AnalysisService(session, ai_provider=provider)
        analysis_run = _start_run(service, incident.public_id)

        completed_run = service.run_core_analysis(analysis_run.id)
        run_id = completed_run.id

    with database_session_factory() as session:
        persisted_run = session.scalar(
            select(AnalysisRun)
            .options(
                selectinload(AnalysisRun.facts),
                selectinload(AnalysisRun.timeline_events),
                selectinload(AnalysisRun.hypotheses),
                selectinload(AnalysisRun.bias_flags),
            )
            .where(AnalysisRun.id == run_id)
        )
        assert persisted_run is not None
        assert persisted_run.status is AnalysisRunStatus.COMPLETED
        assert persisted_run.incident.status is IncidentStatus.COMPLETED
        assert persisted_run.completed_at is not None
        assert persisted_run.error_message is None
        assert persisted_run.provider_name == "fake"
        assert persisted_run.model_name == "fixture-v1"
        assert persisted_run.prompt_versions == {
            "bias": "v1",
            "critic": "v1",
            "system": "v1",
            "summary": "v1",
            "timeline": "v1",
            "hypotheses": "v1",
            "open_questions": "v1",
        }
        assert persisted_run.input_evidence_codes == ["E-001"]
        assert persisted_run.raw_response is not None
        audit_envelope = json.loads(persisted_run.raw_response)
        stages = audit_envelope["stages"]
        assert set(stages) == {
            "summary",
            "timeline",
            "hypotheses",
            "critic",
            "bias",
            "open_questions",
        }

        fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        for stage_name, fixture_name in (
            ("summary", "valid_summary"),
            ("timeline", "valid_timeline"),
            ("hypotheses", "valid_hypotheses"),
            ("critic", "valid_critic"),
            ("bias", "valid_bias"),
            ("open_questions", "valid_open_questions"),
        ):
            assert (
                stages[stage_name]["raw_response"]
                == fixture_bank[fixture_name]["raw_response"]
            )
            assert stages[stage_name]["parsed_output"]
            assert stages[stage_name]["metadata"]["provider_name"] == "fake"
            assert stages[stage_name]["metadata"]["model_name"] == "fixture-v1"

        summary_output = SummaryOutputV1.model_validate(
            stages["summary"]["parsed_output"]
        )
        open_questions_output = OpenQuestionsOutputV1.model_validate(
            stages["open_questions"]["parsed_output"]
        )
        assert all(
            question.evidence_needed for question in open_questions_output.questions
        )
        assert summary_output.assumptions[0].claim == "A deployment may be related."
        assert len(persisted_run.facts) == 1
        assert persisted_run.facts[0].support_status is ClaimSupportStatus.SUPPORTED
        assert persisted_run.facts[0].evidence_codes == ["E-001"]
        assert len(persisted_run.timeline_events) == 2
        direct_event = next(
            event for event in persisted_run.timeline_events if not event.is_inferred
        )
        inferred_event = next(
            event for event in persisted_run.timeline_events if event.is_inferred
        )
        assert direct_event.event_time == datetime(2025, 1, 1, 10, 30, tzinfo=UTC)
        assert inferred_event.event_time is None
        assert inferred_event.confidence == 70
        timeline_output = TimelineOutputV1.model_validate(
            stages["timeline"]["parsed_output"]
        )
        assert timeline_output.events[0].timestamp == "2025-01-01T12:30:00+02:00"
        assert timeline_output.events[1].timestamp == "time unknown"
        assert timeline_output.events[1].uncertainty_explanation == (
            "Only one captured failure is available, so the start time cannot be "
            "established."
        )
        assert len(persisted_run.hypotheses) == 3
        assert sorted(hypothesis.rank for hypothesis in persisted_run.hypotheses) == [
            1,
            2,
            3,
        ]
        assert {risk.bias_type for risk in persisted_run.bias_flags} == {
            "Confirmation bias",
            "Anchoring bias",
            "Automation bias",
            "Post hoc fallacy",
            "Overconfidence bias",
        }

    assert len(provider.requests) == 6
    manifests = [request.evidence_manifest for request in provider.requests]
    assert all(manifest is manifests[0] for manifest in manifests)
    assert all(request.critic_context is None for request in provider.requests[:3])
    assert provider.requests[3].critic_context is not None
    assert provider.requests[4].bias_context is not None
    assert provider.requests[5].open_questions_context is not None
    assert all(secret not in request.model_dump_json() for request in provider.requests)


def test_core_analysis_lowers_confidence_for_valid_contradicting_evidence(
    database_session_factory: sessionmaker[Session],
) -> None:
    fixture_provider = _recording_core_provider()

    def generate_contradicting_hypothesis(
        request: AIRequest,
    ) -> AIResult[AIOutput]:
        result = fixture_provider.generate(request)
        if request.metadata.analysis_stage is not AnalysisStage.HYPOTHESES:
            return result
        output = result.output
        assert isinstance(output, HypothesesOutputV1)
        first_hypothesis = output.hypotheses[0]
        contradiction = ContradictingEvidenceV1(
            reference=EvidenceReferenceV1(
                evidence_id="E-001",
                line_range="2",
                excerpt="database pool healthy",
            ),
            relevance="The observed healthy pool conflicts with pool exhaustion.",
        )
        modified_output = output.model_copy(
            update={
                "hypotheses": (
                    first_hypothesis.model_copy(
                        update={"contradicting_evidence": (contradiction,)},
                    ),
                    *output.hypotheses[1:],
                )
            }
        )
        return AIResult[AIOutput](
            output=modified_output,
            metadata=result.metadata,
            audit=SuccessAuditData(raw_response=modified_output.model_dump_json()),
        )

    provider = _RecordingProvider(generate_contradicting_hypothesis)
    with database_session_factory() as session:
        incident = _persist_incident(
            session,
            original_text="checkout failed\ndatabase pool healthy",
        )
        service = AnalysisService(session, ai_provider=provider)
        analysis_run = _start_run(service, incident.public_id)

        completed_run = service.run_core_analysis(analysis_run.id)
        run_id = completed_run.id

    with database_session_factory() as session:
        persisted_run = session.scalar(
            select(AnalysisRun)
            .options(selectinload(AnalysisRun.hypotheses))
            .where(AnalysisRun.id == run_id)
        )

        assert persisted_run is not None
        top_hypothesis = min(persisted_run.hypotheses, key=lambda item: item.rank)
        assert top_hypothesis.confidence == 50
        assert top_hypothesis.contradicting_evidence_codes == ["E-001"]

        audit_envelope = json.loads(persisted_run.raw_response or "")
        audited_hypothesis = audit_envelope["stages"]["hypotheses"]["parsed_output"][
            "hypotheses"
        ][0]
        assert audited_hypothesis["confidence"] == 60
        assert audited_hypothesis["contradicting_evidence"] == [
            {
                "reference": {
                    "evidence_id": "E-001",
                    "line_range": "2",
                    "excerpt": "database pool healthy",
                },
                "relevance": (
                    "The observed healthy pool conflicts with pool exhaustion."
                ),
            }
        ]


@pytest.mark.parametrize(
    ("reference_update", "expected_evidence_codes"),
    [
        pytest.param(
            {"evidence_id": "E-999"},
            ["E-999"],
            id="unknown-evidence-id",
        ),
        pytest.param(
            {"line_range": "999"},
            ["E-001"],
            id="out-of-range-line-reference",
        ),
    ],
)
def test_core_analysis_flags_invalid_fact_reference_without_failing_run(
    database_session_factory: sessionmaker[Session],
    reference_update: dict[str, str],
    expected_evidence_codes: list[str],
) -> None:
    fixture_provider = _recording_core_provider()

    def generate_invalid_summary_reference(request: AIRequest) -> AIResult[AIOutput]:
        result = fixture_provider.generate(request)
        if request.metadata.analysis_stage is not AnalysisStage.SUMMARY:
            return result
        output = result.output
        assert isinstance(output, SummaryOutputV1)
        fact = output.facts[0]
        invalid_reference = fact.evidence[0].model_copy(update=reference_update)
        return AIResult[AIOutput](
            output=output.model_copy(
                update={
                    "facts": (
                        fact.model_copy(
                            update={"evidence": (invalid_reference,)},
                        ),
                    )
                }
            ),
            metadata=result.metadata,
            audit=result.audit,
        )

    provider = _RecordingProvider(generate_invalid_summary_reference)
    with database_session_factory() as session:
        incident = _persist_incident(session)
        service = AnalysisService(session, ai_provider=provider)
        analysis_run = _start_run(service, incident.public_id)

        completed_run = service.run_core_analysis(analysis_run.id)
        run_id = completed_run.id

    with database_session_factory() as session:
        persisted_run = session.scalar(
            select(AnalysisRun)
            .options(selectinload(AnalysisRun.facts))
            .where(AnalysisRun.id == run_id)
        )

        assert persisted_run is not None
        assert persisted_run.status is AnalysisRunStatus.COMPLETED
        assert len(persisted_run.facts) == 1
        assert persisted_run.facts[0].support_status is ClaimSupportStatus.UNSUPPORTED
        assert persisted_run.facts[0].evidence_codes == expected_evidence_codes
        assert persisted_run.facts[0].supporting_excerpt is None


def test_core_analysis_retains_failed_stage_audit_without_structured_rows(
    database_session_factory: sessionmaker[Session],
) -> None:
    provider = _recording_core_provider(timeline_fixture="invalid_timeline_json")
    with database_session_factory() as session:
        incident = _persist_incident(session)
        service = AnalysisService(session, ai_provider=provider)
        analysis_run = _start_run(service, incident.public_id)
        run_id = analysis_run.id

        with pytest.raises(AIProviderExecutionError) as error_info:
            service.run_core_analysis(run_id)

        assert error_info.value.details.category is AIFailureCategory.MALFORMED_JSON
        failed_raw_response = error_info.value.details.audit
        assert failed_raw_response is not None
        raw_response = failed_raw_response.raw_response
        assert raw_response is not None
        assert raw_response not in str(error_info.value)
        assert raw_response not in repr(error_info.value)

    with database_session_factory() as session:
        persisted_run = session.scalar(
            select(AnalysisRun)
            .options(
                selectinload(AnalysisRun.facts),
                selectinload(AnalysisRun.timeline_events),
                selectinload(AnalysisRun.hypotheses),
            )
            .where(AnalysisRun.id == run_id)
        )
        assert persisted_run is not None
        assert persisted_run.status is AnalysisRunStatus.FAILED
        assert persisted_run.incident.status is IncidentStatus.FAILED
        assert persisted_run.completed_at is not None
        assert persisted_run.error_message == "The AI provider returned malformed JSON."
        assert persisted_run.prompt_versions == {
            "system": "v1",
            "summary": "v1",
            "timeline": "v1",
        }
        assert persisted_run.input_evidence_codes == ["E-001"]
        assert persisted_run.raw_response is not None
        stages = json.loads(persisted_run.raw_response)["stages"]
        assert set(stages) == {"summary", "timeline"}
        assert stages["summary"]["parsed_output"]
        assert stages["timeline"]["failure_category"] == "malformed_json"
        fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        assert (
            stages["timeline"]["raw_response"]
            == fixture_bank["invalid_timeline_json"]["raw_response"]
        )
        assert persisted_run.facts == []
        assert persisted_run.timeline_events == []
        assert persisted_run.hypotheses == []

    assert [request.metadata.analysis_stage for request in provider.requests] == [
        AnalysisStage.SUMMARY,
        AnalysisStage.TIMELINE,
    ]


def test_core_analysis_rolls_back_structured_rows_before_failed_audit_persistence(
    database_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _recording_core_provider()
    completed_flush_failed = False
    with database_session_factory() as session:
        incident = _persist_incident(session)
        service = AnalysisService(session, ai_provider=provider)
        analysis_run = _start_run(service, incident.public_id)
        run_id = analysis_run.id
        original_flush = session.flush

        def fail_first_completed_result_flush() -> None:
            nonlocal completed_flush_failed
            has_pending_structured_rows = any(
                isinstance(item, (Fact, TimelineEvent, Hypothesis))
                for item in session.new
            )
            if has_pending_structured_rows and not completed_flush_failed:
                completed_flush_failed = True
                raise SQLAlchemyError("structured result flush unavailable")
            original_flush()

        flush = Mock(side_effect=fail_first_completed_result_flush)
        monkeypatch.setattr(session, "flush", flush)

        with pytest.raises(AnalysisPersistenceError) as error_info:
            service.run_core_analysis(run_id)

        assert completed_flush_failed is True
        assert len(provider.requests) == 6
        assert str(error_info.value) == (
            "The completed analysis results could not be saved."
        )
        fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        for fixture_name in (
            "valid_summary",
            "valid_timeline",
            "valid_hypotheses",
            "valid_critic",
            "valid_bias",
            "valid_open_questions",
        ):
            raw_response = fixture_bank[fixture_name]["raw_response"]
            assert raw_response not in str(error_info.value)
            assert raw_response not in repr(error_info.value)

    with database_session_factory() as session:
        persisted_run = session.scalar(
            select(AnalysisRun)
            .options(
                selectinload(AnalysisRun.facts),
                selectinload(AnalysisRun.timeline_events),
                selectinload(AnalysisRun.hypotheses),
                selectinload(AnalysisRun.bias_flags),
            )
            .where(AnalysisRun.id == run_id)
        )
        assert persisted_run is not None
        assert persisted_run.status is AnalysisRunStatus.FAILED
        assert persisted_run.incident.status is IncidentStatus.FAILED
        assert persisted_run.error_message == (
            "The completed analysis results could not be saved."
        )
        assert persisted_run.facts == []
        assert persisted_run.timeline_events == []
        assert persisted_run.hypotheses == []
        assert persisted_run.bias_flags == []
        assert persisted_run.prompt_versions == {
            "bias": "v1",
            "critic": "v1",
            "system": "v1",
            "summary": "v1",
            "timeline": "v1",
            "hypotheses": "v1",
            "open_questions": "v1",
        }
        assert persisted_run.input_evidence_codes == ["E-001"]
        assert persisted_run.raw_response is not None
        stages = json.loads(persisted_run.raw_response)["stages"]
        assert set(stages) == {
            "summary",
            "timeline",
            "hypotheses",
            "critic",
            "bias",
            "open_questions",
        }
        for stage_name, fixture_name in (
            ("summary", "valid_summary"),
            ("timeline", "valid_timeline"),
            ("hypotheses", "valid_hypotheses"),
            ("critic", "valid_critic"),
            ("bias", "valid_bias"),
            ("open_questions", "valid_open_questions"),
        ):
            assert (
                stages[stage_name]["raw_response"]
                == fixture_bank[fixture_name]["raw_response"]
            )
            assert stages[stage_name]["parsed_output"]
            assert stages[stage_name]["metadata"]
