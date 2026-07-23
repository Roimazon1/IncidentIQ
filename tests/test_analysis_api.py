"""Endpoint tests for running and reopening the basic analysis workflow."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.config import Settings
from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    ClaimSupportStatus,
    Fact,
    IncidentStatus,
)
from app.routers import analysis as analysis_router
from app.schemas.ai_outputs import (
    AIOutput,
    ContradictingEvidenceV1,
    EvidenceReferenceV1,
    HypothesesOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (
    AIRequest,
    AIResult,
    AnalysisStage,
    SuccessAuditData,
)
from app.services.ai_provider import AIProvider
from app.services.analysis_service import AnalysisService
from app.services.prompt_registry import PromptRegistry
from app.services.providers.fake_provider import FakeAIProvider


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_ai_responses.json"
CORE_FIXTURES = (
    "valid_summary",
    "valid_timeline",
    "valid_hypotheses",
    "valid_critic",
    "valid_bias",
    "valid_open_questions",
)


class _OutOfRangeSummaryProvider:
    def __init__(self, delegate: FakeAIProvider) -> None:
        self._delegate = delegate

    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        result = self._delegate.generate(request)
        if request.metadata.analysis_stage is not AnalysisStage.SUMMARY:
            return result
        output = result.output
        assert isinstance(output, SummaryOutputV1)
        fact = output.facts[0]
        invalid_reference = fact.evidence[0].model_copy(update={"line_range": "999"})
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


class _HighConfidenceTimelineProvider:
    def __init__(self, delegate: FakeAIProvider) -> None:
        self._delegate = delegate

    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        result = self._delegate.generate(request)
        if request.metadata.analysis_stage is not AnalysisStage.TIMELINE:
            return result
        output = result.output
        assert isinstance(output, TimelineOutputV1)
        output_data = output.model_dump()
        output_data["events"][0]["confidence"] = 88
        output_data["events"][1]["confidence"] = 95
        modified_output = TimelineOutputV1.model_validate(output_data)
        return AIResult[AIOutput](
            output=modified_output,
            metadata=result.metadata,
            audit=SuccessAuditData(raw_response=modified_output.model_dump_json()),
        )


class _ContradictingHypothesisProvider:
    def __init__(self, delegate: FakeAIProvider) -> None:
        self._delegate = delegate

    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        result = self._delegate.generate(request)
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


def _configured_service_builder(
    provider: AIProvider,
) -> Callable[[Session, Settings], AnalysisService]:
    def build_service(session: Session, settings: Settings) -> AnalysisService:
        del settings
        return AnalysisService(
            session,
            ai_provider=provider,
            configured_provider_name="fake",
            configured_model_name="fixture-v1",
        )

    return build_service


@pytest.fixture(autouse=True)
def use_fake_analysis_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep application analysis tests independent of local environment values."""
    monkeypatch.setattr(
        analysis_router,
        "settings",
        Settings.model_validate(
            {
                "ai_provider": "fake",
                "gemini_api_key": None,
                "gemini_model": None,
            }
        ),
    )


def _create_ready_incident(
    client: TestClient,
    *,
    name: str = "Checkout failures",
    evidence_text: str = "api_key=local-secret\ncheckout failed",
) -> str:
    create_response = client.post(
        "/incidents",
        data={
            "name": name,
            "description": "Intermittent checkout errors",
            "affected_service": "checkout",
            "reported_start_time": "",
        },
        follow_redirects=False,
    )
    public_id = create_response.headers["location"].split("?")[0].rsplit("/", 1)[-1]
    evidence_response = client.post(
        f"/incidents/{public_id}/evidence/text",
        data={
            "source_name": "checkout.log",
            "original_text": evidence_text,
            "evidence_type": "APPLICATION_LOG",
        },
        follow_redirects=False,
    )
    assert evidence_response.status_code == 303
    return public_id


def test_fake_analysis_can_be_run_and_reopened_without_exposing_raw_audit(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    public_id = _create_ready_incident(database_client)

    start_response = database_client.post(
        f"/incidents/{public_id}/analysis",
        follow_redirects=False,
    )

    assert start_response.status_code == 303
    assert start_response.headers["location"] == (f"/incidents/{public_id}/analysis/1")

    detail_response = database_client.get(start_response.headers["location"])

    assert detail_response.status_code == 200
    assert "COMPLETED" in detail_response.text
    assert "Human review required" in detail_response.text
    assert "AI-generated analysis" in detail_response.text
    assert 'aria-label="Analysis uncertainty key"' in detail_response.text
    assert "Facts: evidence-validated" in detail_response.text
    assert "Assumptions: unverified" in detail_response.text
    assert "Hypotheses: require testing" in detail_response.text
    assert "Actions: proposed, never auto-executed" in detail_response.text
    assert 'aria-label="Analysis sections"' in detail_response.text
    for section_id in (
        "summary-section",
        "evidence-section",
        "facts-assumptions-section",
        "timeline-section",
        "hypotheses-section",
        "next-actions-section",
        "reasoning-risks-section",
        "ai-audit-section",
    ):
        assert f'id="{section_id}"' in detail_response.text
        assert f'href="#{section_id}"' in detail_response.text
    assert "Checkout requests are failing." in detail_response.text
    assert "Review incident evidence" in detail_response.text
    assert (
        'href="http://testserver/incidents/INC-000001/evidence/new?tab=saved"'
        in detail_response.text
    )
    assert "Confirmed facts" in detail_response.text
    assert "The redacted checkout log contains a failure." in detail_response.text
    assert "SUPPORTED" in detail_response.text
    assert "Unconfirmed AI claims" in detail_response.text
    assert "No unconfirmed claims were retained." in detail_response.text
    assert "A deployment may be related." in detail_response.text
    assert "Timeline" in detail_response.text
    assert "Direct" in detail_response.text
    assert "Inferred" in detail_response.text
    assert "Inference uncertainty" in detail_response.text
    assert (
        "Only one captured failure is available, so the start time cannot be "
        "established."
    ) in detail_response.text
    assert "Ranked hypotheses" in detail_response.text
    assert "Database connection pool exhaustion" in detail_response.text
    assert "Recent deployment regression" in detail_response.text
    assert "External payment dependency failure" in detail_response.text
    assert "not confirmed root causes" in detail_response.text
    assert "Adversarial critique" in detail_response.text
    assert "The top hypothesis is only weakly distinguished" in detail_response.text
    assert "does not change the" in detail_response.text
    assert "original hypothesis ranking" in detail_response.text
    assert "critic confidence 35%" in detail_response.text
    assert "Reasoning risks and fallacies" in detail_response.text
    assert "possible risks to investigate, not accusations" in detail_response.text
    assert "Confirmation bias" in detail_response.text
    assert "Anchoring bias" in detail_response.text
    assert "Automation bias" in detail_response.text
    assert "Post hoc fallacy" in detail_response.text
    assert "Overconfidence bias" in detail_response.text
    assert "Actively seek evidence that would weaken" in detail_response.text
    assert "Open questions and evidence needed" in detail_response.text
    assert "unresolved investigation questions, not confirmed facts or causes" in (
        detail_response.text
    )
    assert "Did database pool saturation coincide with checkout failures?" in (
        detail_response.text
    )
    assert "Database pool utilization metrics for the failure window" in (
        detail_response.text
    )
    assert "Missing Evidence: Database pool utilization during the failure window" in (
        detail_response.text
    )
    assert "fake / fixture-v1" in detail_response.text
    assert "summary v1" in detail_response.text
    assert "E-001" in detail_response.text
    assert "Next actions" in detail_response.text
    assert "No recommended actions are available for this analysis run." in (
        detail_response.text
    )
    assert "AI audit" in detail_response.text
    assert "local-secret" not in detail_response.text
    assert '"raw_response"' not in detail_response.text
    assert '"hypotheses":[' not in detail_response.text
    assert "Back to incident INC-000001" in detail_response.text
    assert 'href="http://testserver/incidents/INC-000001"' in detail_response.text

    with database_session_factory() as session:
        persisted_run = session.scalar(
            select(AnalysisRun)
            .options(
                selectinload(AnalysisRun.facts),
                selectinload(AnalysisRun.timeline_events),
                selectinload(AnalysisRun.hypotheses),
                selectinload(AnalysisRun.bias_flags),
            )
            .where(AnalysisRun.id == 1)
        )
        assert persisted_run is not None
        assert persisted_run.status is AnalysisRunStatus.COMPLETED
        assert persisted_run.incident.status is IncidentStatus.COMPLETED
        assert len(persisted_run.facts) == 1
        assert persisted_run.facts[0].support_status is ClaimSupportStatus.SUPPORTED
        assert len(persisted_run.timeline_events) == 2
        assert len(persisted_run.hypotheses) == 3
        assert len(persisted_run.bias_flags) == 5
        assert [
            (hypothesis.rank, hypothesis.title, hypothesis.confidence)
            for hypothesis in persisted_run.hypotheses
        ] == [
            (1, "Database connection pool exhaustion", 60),
            (2, "Recent deployment regression", 45),
            (3, "External payment dependency failure", 35),
        ]


def test_inferred_provider_confidence_is_capped_after_auditing(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = PromptRegistry()
    provider = _HighConfidenceTimelineProvider(
        FakeAIProvider.from_file_set(
            FIXTURE_PATH,
            CORE_FIXTURES,
            prompt_resolver=registry.resolve_content,
            prompt_bundle_validator=registry.validate_bundle,
        )
    )
    monkeypatch.setattr(
        analysis_router,
        "build_configured_analysis_service",
        _configured_service_builder(provider),
    )
    public_id = _create_ready_incident(database_client)

    start_response = database_client.post(
        f"/incidents/{public_id}/analysis",
        follow_redirects=False,
    )
    detail_response = database_client.get(start_response.headers["location"])

    with database_session_factory() as session:
        persisted_run = session.scalar(
            select(AnalysisRun).options(selectinload(AnalysisRun.timeline_events))
        )
        assert persisted_run is not None
        assert persisted_run.status is AnalysisRunStatus.COMPLETED
        direct_event = next(
            event for event in persisted_run.timeline_events if not event.is_inferred
        )
        inferred_event = next(
            event for event in persisted_run.timeline_events if event.is_inferred
        )
        assert direct_event.confidence == 88
        assert inferred_event.confidence == 70

        audit_envelope = json.loads(persisted_run.raw_response or "")
        timeline_audit = audit_envelope["stages"]["timeline"]
        audited_events = timeline_audit["parsed_output"]["events"]
        raw_events = json.loads(timeline_audit["raw_response"])["events"]
        assert [event["confidence"] for event in audited_events] == [88, 95]
        assert [event["confidence"] for event in raw_events] == [88, 95]

    assert start_response.status_code == 303
    assert detail_response.status_code == 200
    assert "Confidence 88%" in detail_response.text
    assert "Confidence 70%" in detail_response.text
    assert "Confidence 95%" not in detail_response.text
    assert "Inference uncertainty" in detail_response.text
    assert (
        "Only one captured failure is available, so the start time cannot be "
        "established."
    ) in detail_response.text
    assert '"raw_response"' not in detail_response.text


def test_out_of_range_fact_is_separate_from_confirmed_facts(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = PromptRegistry()
    provider = _OutOfRangeSummaryProvider(
        FakeAIProvider.from_file_set(
            FIXTURE_PATH,
            CORE_FIXTURES,
            prompt_resolver=registry.resolve_content,
            prompt_bundle_validator=registry.validate_bundle,
        )
    )

    monkeypatch.setattr(
        analysis_router,
        "build_configured_analysis_service",
        _configured_service_builder(provider),
    )
    public_id = _create_ready_incident(database_client)
    start_response = database_client.post(
        f"/incidents/{public_id}/analysis",
        follow_redirects=False,
    )

    with database_session_factory() as session:
        fact = session.scalar(select(Fact))
        assert fact is not None
        assert fact.support_status is ClaimSupportStatus.UNSUPPORTED
        assert fact.supporting_excerpt is None

    detail_response = database_client.get(start_response.headers["location"])
    claim = "The redacted checkout log contains a failure."

    assert detail_response.status_code == 200
    assert "Confirmed facts" in detail_response.text
    assert "No AI claims have validated support." in detail_response.text
    assert "Unconfirmed AI claims" in detail_response.text
    assert "UNSUPPORTED" in detail_response.text
    assert detail_response.text.count(claim) == 1
    assert detail_response.text.index(claim) > detail_response.text.index(
        "Unconfirmed AI claims"
    )


def test_valid_contradiction_remains_visible_with_adjusted_confidence(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = PromptRegistry()
    provider = _ContradictingHypothesisProvider(
        FakeAIProvider.from_file_set(
            FIXTURE_PATH,
            CORE_FIXTURES,
            prompt_resolver=registry.resolve_content,
            prompt_bundle_validator=registry.validate_bundle,
        )
    )
    monkeypatch.setattr(
        analysis_router,
        "build_configured_analysis_service",
        _configured_service_builder(provider),
    )
    public_id = _create_ready_incident(
        database_client,
        evidence_text="checkout failed\ndatabase pool healthy",
    )

    start_response = database_client.post(
        f"/incidents/{public_id}/analysis",
        follow_redirects=False,
    )
    detail_response = database_client.get(start_response.headers["location"])

    with database_session_factory() as session:
        persisted_run = session.scalar(
            select(AnalysisRun).options(selectinload(AnalysisRun.hypotheses))
        )
        assert persisted_run is not None
        top_hypothesis = min(persisted_run.hypotheses, key=lambda item: item.rank)
        assert top_hypothesis.confidence == 50
        assert top_hypothesis.contradicting_evidence_codes == ["E-001"]

    assert detail_response.status_code == 200
    assert "Confidence 50%" in detail_response.text
    assert "Contradicting evidence" in detail_response.text
    assert "E-001" in detail_response.text
    assert "Confidence is reduced deterministically" in detail_response.text
    assert '"raw_response"' not in detail_response.text


def test_running_analysis_renders_pending_page(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    public_id = _create_ready_incident(database_client)
    with database_session_factory() as session:
        analysis_run = AnalysisService(session).start_analysis_run(
            public_id,
            provider_name="fake",
            model_name="fixture-v1",
        )
        run_id = analysis_run.id

    response = database_client.get(f"/incidents/{public_id}/analysis/{run_id}")

    assert response.status_code == 200
    assert "RUNNING" in response.text
    assert "still running" in response.text
    assert "fixture-v1" in response.text
    assert "raw_response" not in response.text
    assert f"Back to incident {public_id}" in response.text
    assert f'href="http://testserver/incidents/{public_id}"' in response.text


def test_failed_fake_analysis_remains_visible_without_raw_response(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = PromptRegistry()
    provider = FakeAIProvider.from_file_set(
        FIXTURE_PATH,
        ("valid_summary", "invalid_timeline_json", "valid_hypotheses"),
        prompt_resolver=registry.resolve_content,
        prompt_bundle_validator=registry.validate_bundle,
    )

    def build_failed_analysis_service(
        session: Session,
        settings: Settings,
    ) -> AnalysisService:
        del settings
        return AnalysisService(
            session,
            ai_provider=provider,
            configured_provider_name="fake",
            configured_model_name="fixture-v1",
        )

    monkeypatch.setattr(
        analysis_router,
        "build_configured_analysis_service",
        build_failed_analysis_service,
    )
    public_id = _create_ready_incident(database_client)

    start_response = database_client.post(
        f"/incidents/{public_id}/analysis",
        follow_redirects=False,
    )
    detail_response = database_client.get(start_response.headers["location"])

    assert start_response.status_code == 303
    assert detail_response.status_code == 200
    assert "FAILED" in detail_response.text
    assert "Analysis failed safely" in detail_response.text
    assert "The AI provider returned malformed JSON." in detail_response.text
    assert "Checkout requests are failing." in detail_response.text
    assert '{"events":' not in detail_response.text
    assert '"raw_response"' not in detail_response.text

    with database_session_factory() as session:
        persisted_run = session.scalar(
            select(AnalysisRun)
            .options(
                selectinload(AnalysisRun.facts),
                selectinload(AnalysisRun.timeline_events),
                selectinload(AnalysisRun.hypotheses),
            )
            .where(AnalysisRun.id == 1)
        )
        assert persisted_run is not None
        assert persisted_run.status is AnalysisRunStatus.FAILED
        assert persisted_run.incident.status is IncidentStatus.FAILED
        assert persisted_run.facts == []
        assert persisted_run.timeline_events == []
        assert persisted_run.hypotheses == []
        assert persisted_run.raw_response is not None


def test_analysis_route_rejects_incident_without_evidence(
    database_client: TestClient,
) -> None:
    create_response = database_client.post(
        "/incidents",
        data={
            "name": "No evidence",
            "description": "Nothing has been supplied yet",
            "affected_service": "checkout",
            "reported_start_time": "",
        },
        follow_redirects=False,
    )
    public_id = create_response.headers["location"].split("?")[0].rsplit("/", 1)[-1]

    response = database_client.post(f"/incidents/{public_id}/analysis")

    assert response.status_code == 409
    assert response.json() == {
        "detail": f"Incident {public_id} requires evidence before analysis."
    }


def test_analysis_route_returns_safe_conflict_for_existing_running_run(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    public_id = _create_ready_incident(database_client)
    with database_session_factory() as session:
        AnalysisService(session).start_analysis_run(
            public_id,
            provider_name="fake",
            model_name="fixture-v1",
        )

    response = database_client.post(f"/incidents/{public_id}/analysis")

    assert response.status_code == 409
    assert response.json() == {
        "detail": f"Incident {public_id} already has a running analysis."
    }


def test_analysis_run_is_scoped_to_its_incident(
    database_client: TestClient,
) -> None:
    first_public_id = _create_ready_incident(database_client)
    second_public_id = _create_ready_incident(
        database_client,
        name="Second incident",
    )
    start_response = database_client.post(
        f"/incidents/{first_public_id}/analysis",
        follow_redirects=False,
    )
    run_id = start_response.headers["location"].rsplit("/", 1)[-1]

    response = database_client.get(f"/incidents/{second_public_id}/analysis/{run_id}")

    assert response.status_code == 404
    assert response.json() == {
        "detail": (
            f"Analysis run {run_id} was not found for incident {second_public_id}."
        )
    }
