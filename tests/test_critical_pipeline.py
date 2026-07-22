"""Focused integration coverage for the separate adversarial critic pass."""

import json
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    ClaimSupportStatus,
    EvidenceItem,
    EvidenceType,
    Incident,
    IncidentStatus,
)
from app.schemas.ai_outputs import (
    AIOutput,
    CriticOutputV1,
    HypothesesOutputV1,
    OpenQuestionsOutputV1,
    ReasoningRisksOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (
    AIRequest,
    AIResult,
    AnalysisStage,
    EvidenceReferenceValidationStatus,
    SuccessAuditData,
)
from app.services.analysis_service import AnalysisService
from app.services.analysis_stage_runner import (
    AnalysisStageOutputError,
    AnalysisStageRunner,
)
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


class RecordingFakeProvider:
    """Record provider-safe requests while delegating to the offline fake."""

    def __init__(
        self,
        provider: FakeAIProvider,
        *,
        replacement_top_hypothesis: str | None = None,
        validation_scenario: bool = False,
    ) -> None:
        self._provider = provider
        self._replacement_top_hypothesis = replacement_top_hypothesis
        self._validation_scenario = validation_scenario
        self.requests: list[AIRequest] = []

    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        self.requests.append(request)
        result = self._provider.generate(request)
        if (
            request.metadata.analysis_stage is AnalysisStage.HYPOTHESES
            and self._replacement_top_hypothesis is not None
        ):
            assert isinstance(result.output, HypothesesOutputV1)
            output_data = result.output.model_dump()
            output_data["hypotheses"][0]["title"] = self._replacement_top_hypothesis
            return AIResult[AIOutput](
                output=HypothesesOutputV1.model_validate(output_data),
                metadata=result.metadata,
                audit=result.audit,
            )
        if self._validation_scenario:
            output = self._validation_scenario_output(request, result.output)
            if output is not result.output:
                return AIResult[AIOutput](
                    output=output,
                    metadata=result.metadata,
                    audit=SuccessAuditData(raw_response=output.model_dump_json()),
                )
        return result

    @staticmethod
    def _validation_scenario_output(
        request: AIRequest,
        output: AIOutput,
    ) -> AIOutput:
        output_data = output.model_dump()
        if request.metadata.analysis_stage is AnalysisStage.SUMMARY:
            assert isinstance(output, SummaryOutputV1)
            output_data["facts"][0]["evidence"][0]["evidence_id"] = "E-999"
            return SummaryOutputV1.model_validate(output_data)
        if request.metadata.analysis_stage is AnalysisStage.TIMELINE:
            assert isinstance(output, TimelineOutputV1)
            output_data["events"][1]["confidence"] = 95
            return TimelineOutputV1.model_validate(output_data)
        if request.metadata.analysis_stage is AnalysisStage.HYPOTHESES:
            assert isinstance(output, HypothesesOutputV1)
            output_data["hypotheses"][0]["contradicting_evidence"] = [
                {
                    "reference": {
                        "evidence_id": "E-001",
                        "line_range": "1-2",
                    },
                    "relevance": "The captured failure may have another cause.",
                }
            ]
            output_data["hypotheses"][1]["contradicting_evidence"] = [
                {
                    "reference": {
                        "evidence_id": "E-999",
                        "line_range": "1",
                    },
                    "relevance": "This generated reference is invalid.",
                }
            ]
            return HypothesesOutputV1.model_validate(output_data)
        return output


def _recording_provider(
    *,
    replacement_top_hypothesis: str | None = None,
    validation_scenario: bool = False,
) -> RecordingFakeProvider:
    registry = PromptRegistry()
    fake_provider = FakeAIProvider.from_file_set(
        FIXTURE_PATH,
        CORE_FIXTURES,
        prompt_resolver=registry.resolve_content,
        prompt_bundle_validator=registry.validate_bundle,
    )
    return RecordingFakeProvider(
        fake_provider,
        replacement_top_hypothesis=replacement_top_hypothesis,
        validation_scenario=validation_scenario,
    )


def _persist_incident(session: Session, public_id: str, secret: str) -> Incident:
    incident = Incident(
        public_id=public_id,
        name="Checkout failures",
        description="Intermittent checkout errors",
        affected_service="checkout",
        status=IncidentStatus.READY,
    )
    incident.evidence_items.append(
        EvidenceItem(
            evidence_code="E-001",
            source_name="checkout.log",
            evidence_type=EvidenceType.APPLICATION_LOG,
            original_text=f"api_key={secret}\ncheckout failed",
            checksum="a" * 64,
        )
    )
    session.add(incident)
    session.commit()
    return incident


def test_critic_challenges_top_hypothesis_without_mutating_original_results(
    database_session_factory: sessionmaker[Session],
) -> None:
    secret = "critic-pipeline-secret"
    provider = _recording_provider()

    with database_session_factory() as session:
        incident = _persist_incident(session, "INC-000001", secret)

        service = AnalysisService(session, ai_provider=provider)
        analysis_run = service.start_analysis_run(
            incident.public_id,
            provider_name="fake",
            model_name="fixture-v1",
        )
        service.run_core_analysis(analysis_run.id)

        session.expire_all()
        persisted_run = session.scalar(
            select(AnalysisRun)
            .options(
                selectinload(AnalysisRun.hypotheses),
                selectinload(AnalysisRun.bias_flags),
            )
            .where(AnalysisRun.id == analysis_run.id)
        )

        assert persisted_run is not None
        assert persisted_run.status is AnalysisRunStatus.COMPLETED
        assert [
            (hypothesis.rank, hypothesis.title, hypothesis.confidence)
            for hypothesis in persisted_run.hypotheses
        ] == [
            (1, "Database connection pool exhaustion", 60),
            (2, "Recent deployment regression", 45),
            (3, "External payment dependency failure", 35),
        ]

        audit_envelope = json.loads(persisted_run.raw_response or "")
        critic_record = audit_envelope["stages"][AnalysisStage.CRITIC.value]
        critic_output = CriticOutputV1.model_validate(critic_record["parsed_output"])

        assert critic_output.findings[0].affected_claim == (
            "Database connection pool exhaustion"
        )
        assert critic_output.alternative_hypothesis is not None
        assert critic_output.alternative_hypothesis.hypothesis_id == "H-003"
        assert critic_record["metadata"]["analysis_stage"] == "critic"
        assert critic_record["metadata"]["task_prompt"] == {
            "name": "critic",
            "version": "v1",
        }
        assert critic_record["metadata"]["output_schema"] == "critic_v1"
        bias_record = audit_envelope["stages"][AnalysisStage.BIAS.value]
        bias_output = ReasoningRisksOutputV1.model_validate(
            bias_record["parsed_output"]
        )
        assert len(bias_output.risks) == 5
        assert len(persisted_run.bias_flags) == 5
        assert bias_record["metadata"]["analysis_stage"] == "bias"
        assert bias_record["metadata"]["task_prompt"] == {
            "name": "bias",
            "version": "v1",
        }
        assert bias_record["metadata"]["output_schema"] == "reasoning_risks_v1"
        open_questions_record = audit_envelope["stages"][
            AnalysisStage.OPEN_QUESTIONS.value
        ]
        assert len(open_questions_record["parsed_output"]["questions"]) == 3
        assert open_questions_record["metadata"]["output_schema"] == (
            "open_questions_v1"
        )

    assert [request.metadata.analysis_stage for request in provider.requests] == [
        AnalysisStage.SUMMARY,
        AnalysisStage.TIMELINE,
        AnalysisStage.HYPOTHESES,
        AnalysisStage.CRITIC,
        AnalysisStage.BIAS,
        AnalysisStage.OPEN_QUESTIONS,
    ]
    manifests = [request.evidence_manifest for request in provider.requests]
    assert all(manifest is manifests[0] for manifest in manifests)
    assert all(request.critic_context is None for request in provider.requests[:3])
    critic_request = provider.requests[3]
    assert critic_request.critic_context is not None
    assert critic_request.critic_context.summary.summary.text == (
        "Checkout requests are failing."
    )
    assert critic_request.critic_context.timeline.events[0].description == (
        "The checkout log records a failed request."
    )
    assert critic_request.critic_context.hypotheses.hypotheses[0].title == (
        "Database connection pool exhaustion"
    )
    bias_request = provider.requests[4]
    assert bias_request.bias_context is not None
    assert bias_request.bias_context.original_analysis == critic_request.critic_context
    assert bias_request.bias_context.validated_analysis.facts[0].support_status is (
        ClaimSupportStatus.SUPPORTED
    )
    assert bias_request.bias_context.critic.findings[0].affected_claim == (
        "Database connection pool exhaustion"
    )
    open_questions_request = provider.requests[5]
    assert open_questions_request.open_questions_context is not None
    assert (
        open_questions_request.open_questions_context.analysis_context
        == bias_request.bias_context
    )
    assert (
        open_questions_request.open_questions_context.reasoning_risks.risks[0].name
        == "Confirmation bias"
    )
    assert '"raw_response"' not in critic_request.model_dump_json()
    assert '"audit"' not in critic_request.model_dump_json()
    assert all(secret not in request.model_dump_json() for request in provider.requests)
    assert all(
        "[REDACTED_API_KEY]" in request.model_dump_json()
        for request in provider.requests
    )


def test_changed_top_hypothesis_changes_typed_critic_request_context(
    database_session_factory: sessionmaker[Session],
) -> None:
    original_provider = _recording_provider()
    changed_provider = _recording_provider(
        replacement_top_hypothesis="Cache saturation",
    )

    with database_session_factory() as session:
        original_incident = _persist_incident(
            session,
            "INC-000001",
            "original-secret",
        )
        original_service = AnalysisService(session, ai_provider=original_provider)
        original_run = original_service.start_analysis_run(
            original_incident.public_id,
            provider_name="fake",
            model_name="fixture-v1",
        )
        original_service.run_core_analysis(original_run.id)

        changed_incident = _persist_incident(
            session,
            "INC-000002",
            "changed-secret",
        )
        changed_service = AnalysisService(session, ai_provider=changed_provider)
        changed_run = changed_service.start_analysis_run(
            changed_incident.public_id,
            provider_name="fake",
            model_name="fixture-v1",
        )
        changed_service.run_core_analysis(changed_run.id)

    original_critic_request = next(
        request
        for request in original_provider.requests
        if request.metadata.analysis_stage is AnalysisStage.CRITIC
    )
    changed_critic_request = next(
        request
        for request in changed_provider.requests
        if request.metadata.analysis_stage is AnalysisStage.CRITIC
    )
    original_context = original_critic_request.critic_context
    changed_context = changed_critic_request.critic_context

    assert original_context is not None
    assert changed_context is not None
    assert original_context.hypotheses.hypotheses[0].title == (
        "Database connection pool exhaustion"
    )
    assert changed_context.hypotheses.hypotheses[0].title == "Cache saturation"
    assert changed_context != original_context
    changed_bias_request = next(
        request
        for request in changed_provider.requests
        if request.metadata.analysis_stage is AnalysisStage.BIAS
    )
    assert changed_bias_request.bias_context is not None
    assert (
        changed_bias_request.bias_context.original_analysis.hypotheses.hypotheses[
            0
        ].title
        == "Cache saturation"
    )


def test_bias_receives_the_same_deterministic_view_used_for_persistence(
    database_session_factory: sessionmaker[Session],
) -> None:
    secret = "validated-bias-context-secret"
    provider = _recording_provider(validation_scenario=True)

    with database_session_factory() as session:
        incident = _persist_incident(session, "INC-000001", secret)
        service = AnalysisService(session, ai_provider=provider)
        analysis_run = service.start_analysis_run(
            incident.public_id,
            provider_name="fake",
            model_name="fixture-v1",
        )

        service.run_core_analysis(analysis_run.id)

        session.expire_all()
        persisted_run = session.scalar(
            select(AnalysisRun)
            .options(
                selectinload(AnalysisRun.facts),
                selectinload(AnalysisRun.timeline_events),
                selectinload(AnalysisRun.hypotheses),
            )
            .where(AnalysisRun.id == analysis_run.id)
        )
        assert persisted_run is not None
        assert persisted_run.status is AnalysisRunStatus.COMPLETED
        assert persisted_run.facts[0].support_status is ClaimSupportStatus.UNSUPPORTED
        assert (
            next(
                event.confidence
                for event in persisted_run.timeline_events
                if event.is_inferred
            )
            == 70
        )
        persisted_confidence = {
            hypothesis.rank: hypothesis.confidence
            for hypothesis in persisted_run.hypotheses
        }
        assert persisted_confidence[1] == 50
        assert persisted_confidence[2] == 45

        audit_envelope = json.loads(persisted_run.raw_response or "")
        summary_audit = audit_envelope["stages"]["summary"]
        timeline_audit = audit_envelope["stages"]["timeline"]
        hypotheses_audit = audit_envelope["stages"]["hypotheses"]
        assert (
            summary_audit["parsed_output"]["facts"][0]["evidence"][0]["evidence_id"]
            == "E-999"
        )
        assert timeline_audit["parsed_output"]["events"][1]["confidence"] == 95
        assert (
            json.loads(timeline_audit["raw_response"])["events"][1]["confidence"] == 95
        )
        assert [
            hypothesis["confidence"]
            for hypothesis in hypotheses_audit["parsed_output"]["hypotheses"][:2]
        ] == [60, 45]

    bias_request = next(
        request
        for request in provider.requests
        if request.metadata.analysis_stage is AnalysisStage.BIAS
    )
    context = bias_request.bias_context
    assert context is not None
    assert context.original_analysis.summary.facts[0].evidence[0].evidence_id == (
        "E-999"
    )
    assert context.original_analysis.timeline.events[1].confidence == 95
    assert context.original_analysis.hypotheses.hypotheses[0].confidence == 60
    assert context.original_analysis.hypotheses.hypotheses[1].confidence == 45

    validated_fact = context.validated_analysis.facts[0]
    assert validated_fact.support_status is ClaimSupportStatus.UNSUPPORTED
    assert validated_fact.evidence[0].status is (
        EvidenceReferenceValidationStatus.UNKNOWN_EVIDENCE_ID
    )
    inferred_event = next(
        event for event in context.validated_analysis.timeline if event.is_inferred
    )
    assert inferred_event.persisted_confidence == 70
    assert inferred_event.uncertainty_explanation == (
        "Only one captured failure is available, so the start time cannot be "
        "established."
    )
    valid_contradiction = context.validated_analysis.hypotheses[0]
    assert valid_contradiction.adjusted_confidence == 50
    assert valid_contradiction.contradicting_evidence[0].reference.status is (
        EvidenceReferenceValidationStatus.VALID
    )
    invalid_contradiction = context.validated_analysis.hypotheses[1]
    assert invalid_contradiction.adjusted_confidence == 45
    assert invalid_contradiction.contradicting_evidence[0].reference.status is (
        EvidenceReferenceValidationStatus.UNKNOWN_EVIDENCE_ID
    )
    assert context.critic.findings[0].affected_claim == (
        "Database connection pool exhaustion"
    )

    serialized_request = bias_request.model_dump_json()
    assert secret not in serialized_request
    assert "raw_response" not in serialized_request
    assert "audit" not in serialized_request

    open_questions_request = next(
        request
        for request in provider.requests
        if request.metadata.analysis_stage is AnalysisStage.OPEN_QUESTIONS
    )
    open_context = open_questions_request.open_questions_context
    assert open_context is not None
    assert open_context.analysis_context == context
    assert open_context.reasoning_risks.risks[0].name == "Confirmation bias"
    serialized_open_request = open_questions_request.model_dump_json()
    assert secret not in serialized_open_request
    assert "raw_response" not in serialized_open_request
    assert "audit" not in serialized_open_request

    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    invalid_output_data = json.loads(
        fixture_bank["valid_open_questions"]["raw_response"]
    )
    invalid_output_data["questions"][0]["source_reference"] = "Invented unresolved item"
    with pytest.raises(
        AnalysisStageOutputError,
        match="untraceable analysis source",
    ) as error_info:
        AnalysisStageRunner.require_traceable_open_questions(
            OpenQuestionsOutputV1.model_validate(invalid_output_data),
            open_context,
            raw_response="internal-sensitive-response",
        )
    assert "internal-sensitive-response" not in str(error_info.value)
    assert error_info.value.audit_raw_response == "internal-sensitive-response"
