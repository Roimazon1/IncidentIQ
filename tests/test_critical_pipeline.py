"""Focused integration coverage for the separate adversarial critic pass."""

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    EvidenceItem,
    EvidenceType,
    Incident,
    IncidentStatus,
)
from app.schemas.ai_outputs import AIOutput, CriticOutputV1, HypothesesOutputV1
from app.schemas.ai_provider import AIRequest, AIResult, AnalysisStage
from app.services.analysis_service import AnalysisService
from app.services.prompt_registry import PromptRegistry
from app.services.providers.fake_provider import FakeAIProvider


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_ai_responses.json"
CORE_FIXTURES = (
    "valid_summary",
    "valid_timeline",
    "valid_hypotheses",
    "valid_critic",
)


class RecordingFakeProvider:
    """Record provider-safe requests while delegating to the offline fake."""

    def __init__(
        self,
        provider: FakeAIProvider,
        *,
        replacement_top_hypothesis: str | None = None,
    ) -> None:
        self._provider = provider
        self._replacement_top_hypothesis = replacement_top_hypothesis
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
        return result


def _recording_provider(
    *,
    replacement_top_hypothesis: str | None = None,
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
            .options(selectinload(AnalysisRun.hypotheses))
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

    assert [request.metadata.analysis_stage for request in provider.requests] == [
        AnalysisStage.SUMMARY,
        AnalysisStage.TIMELINE,
        AnalysisStage.HYPOTHESES,
        AnalysisStage.CRITIC,
    ]
    manifests = [request.evidence_manifest for request in provider.requests]
    assert manifests[0] is manifests[1] is manifests[2] is manifests[3]
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

    original_context = original_provider.requests[-1].critic_context
    changed_context = changed_provider.requests[-1].critic_context

    assert original_context is not None
    assert changed_context is not None
    assert original_context.hypotheses.hypotheses[0].title == (
        "Database connection pool exhaustion"
    )
    assert changed_context.hypotheses.hypotheses[0].title == "Cache saturation"
    assert changed_context != original_context
