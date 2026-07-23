"""Focused regressions for reviewed, secret-safe report input."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    ClaimSupportStatus,
    EvidenceItem,
    EvidenceType,
    Fact,
    FactReviewStatus,
    HypothesisStatus,
    Incident,
    IncidentStatus,
    RecommendedAction,
    Report,
)
from app.schemas.ai_outputs import AIOutput, PostmortemOutputV1
from app.schemas.ai_provider import (
    AIFailureCategory,
    AIRequest,
    AIResult,
    EvidenceReferenceValidationStatus,
)
from app.schemas.report import (
    ReportAssumptionSource,
    ReportFactCategory,
)
from app.schemas.review import (
    FactReviewUpdate,
    HumanNoteCreate,
    HypothesisReviewUpdate,
    TimelineReviewUpdate,
)
from app.services.ai_provider import (
    build_ai_result,
    raise_ai_provider_failure,
    resolve_request_prompts,
)
from app.services.analysis_service import AnalysisRunNotFoundError, AnalysisService
from app.services.prompt_registry import PromptRegistry
from app.services.providers.fake_provider import FakeAIProvider
from app.services.providers.gemini_provider import GeminiAIProvider
from app.services.report_service import (
    POSTMORTEM_SECTIONS,
    ReportGenerationError,
    ReportInputUnavailableError,
    ReportPersistenceError,
    ReportProviderExecutionError,
    ReportService,
)
from app.services.review_service import ReviewService


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_ai_responses.json"
ORIGINAL_EVIDENCE_SECRET = "api_key=original-report-secret"
LATER_EVIDENCE_SECRET = "password=later-report-secret"
RAW_PROVIDER_SECRET = "raw-provider-report-secret"
GENERATED_OUTPUT_SECRET = "generated-output-secret"


@dataclass(frozen=True, slots=True)
class ReviewedReportAnalysis:
    public_id: str
    run_id: int
    accepted_fact_id: int
    unsupported_fact_id: int
    reclassified_fact_id: int
    inferred_event_id: int
    reviewed_hypothesis_id: int
    raw_response: str


class _RecordingPostmortemProvider:
    provider_name = "fake"
    model_name = "fixture-v1"

    def __init__(self, output: PostmortemOutputV1) -> None:
        self.output = output
        self.requests: list[AIRequest] = []
        self._prompt_registry = PromptRegistry()

    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        resolve_request_prompts(
            request,
            prompt_resolver=self._prompt_registry.resolve_content,
            prompt_bundle_validator=self._prompt_registry.validate_bundle,
        )
        self.requests.append(request)
        return build_ai_result(
            request=request,
            output=self.output,
            provider_name=self.provider_name,
            model_name=self.model_name,
            attempt_count=1,
            raw_response=self.output.model_dump_json(),
        )


class _FailingPostmortemProvider:
    provider_name = "fake"
    model_name = "fixture-v1"

    def __init__(self, category: AIFailureCategory) -> None:
        self.category = category
        self.requests: list[AIRequest] = []

    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        self.requests.append(request)
        raise_ai_provider_failure(
            request=request,
            category=self.category,
            attempt_count=1,
            raw_response=f'{{"private":"{RAW_PROVIDER_SECRET}"}}',
        )


class _MismatchedPostmortemProvider(_RecordingPostmortemProvider):
    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        result = super().generate(request)
        return result.model_copy(
            update={
                "metadata": result.metadata.model_copy(
                    update={"request_identifier": "wrong-request"}
                )
            }
        )


def _postmortem_output() -> PostmortemOutputV1:
    return PostmortemOutputV1(
        executive_summary=(
            "Checkout failures remain under human review; the root cause is not "
            "yet verified."
        ),
        incident_impact="Some customers could not complete checkout.",
        detection="Detection timing is unknown and requires human follow-up.",
        evidence_reviewed="Reviewed E-001; E-002 and E-999 remain unavailable.",
        timeline=(
            "The direct E-001 event is supported; the earlier onset is possible "
            "but inferred."
        ),
        confirmed_facts="The E-001 checkout failure was accepted by a human.",
        assumptions_and_unresolved_questions=(
            "Deployment causation remains an assumption. Database saturation is "
            "not yet verified."
        ),
        root_cause_hypotheses_and_confidence=(
            "Database pool exhaustion is supported by a human at 73% confidence, "
            "not confirmed."
        ),
        supporting_and_contradicting_evidence=(
            "Valid E-001 references support and weaken the leading hypothesis; "
            "invalid references are excluded."
        ),
        investigation_actions=(
            "A human may compare telemetry; IncidentIQ executes no command."
        ),
        mitigation_and_recovery="Mitigation and recovery are not yet known.",
        biases_and_reasoning_risks=(
            "Anchoring and automation bias remain possible reasoning risks."
        ),
        ai_limitations_and_unsupported_claims=(
            "The E-999 claim is unsupported and unavailable evidence limits the AI."
        ),
        lessons_learned=f"Never store api_key={GENERATED_OUTPUT_SECRET} in reports.",
        follow_up_actions="<script>Human approval is required.</script>",
    )


def _core_provider() -> FakeAIProvider:
    registry = PromptRegistry()
    return FakeAIProvider.from_file_set(
        FIXTURE_PATH,
        (
            "valid_summary",
            "valid_timeline",
            "valid_hypotheses",
            "valid_critic",
            "valid_bias",
            "valid_open_questions",
        ),
        prompt_resolver=registry.resolve_content,
        prompt_bundle_validator=registry.validate_bundle,
    )


def _create_reviewed_report_analysis(session: Session) -> ReviewedReportAnalysis:
    incident = Incident(
        public_id="INC-000001",
        name="Reviewed checkout incident",
        description="A report-service fixture.",
        affected_service="checkout",
        status=IncidentStatus.READY,
        evidence_items=[
            EvidenceItem(
                evidence_code="E-001",
                source_name="checkout.log",
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text=f"{ORIGINAL_EVIDENCE_SECRET}\ncheckout failed",
                redacted_text="api_key=[REDACTED]\ncheckout failed",
                checksum="a" * 64,
            )
        ],
    )
    session.add(incident)
    session.commit()

    analysis_service = AnalysisService(session, ai_provider=_core_provider())
    analysis_run = analysis_service.start_analysis_run(
        incident.public_id,
        provider_name="fake",
        model_name="fixture-v1",
    )
    analysis_service.run_core_analysis(analysis_run.id)
    session.refresh(analysis_run)

    accepted_fact = analysis_run.facts[0]
    unsupported_fact = Fact(
        claim="An unavailable response proves the payment provider failed.",
        support_status=ClaimSupportStatus.UNSUPPORTED,
        confidence=84,
        evidence_codes=["E-999"],
        analysis_run=analysis_run,
    )
    reclassified_fact = Fact(
        claim="The deployment caused the checkout failure.",
        support_status=ClaimSupportStatus.SUPPORTED,
        confidence=76,
        evidence_codes=["E-001"],
        analysis_run=analysis_run,
    )
    action = RecommendedAction(
        analysis_run=analysis_run,
        description="Compare database telemetry with checkout failures.",
        priority="HIGH",
        evidence_codes=["E-001", "E-002"],
        owner_role="Incident commander",
        expected_information="Whether pool pressure aligns with failures.",
        operational_risk="Read-only investigation; no command is executed.",
    )
    action.hypotheses.append(analysis_run.hypotheses[0])
    session.add_all([unsupported_fact, reclassified_fact, action])
    session.commit()

    review_service = ReviewService(session)
    review_service.review_fact(
        incident.public_id,
        analysis_run.id,
        accepted_fact.id,
        FactReviewUpdate(decision=FactReviewStatus.ACCEPTED),
    )
    review_service.review_fact(
        incident.public_id,
        analysis_run.id,
        unsupported_fact.id,
        FactReviewUpdate(decision=FactReviewStatus.REJECTED),
    )
    review_service.review_fact(
        incident.public_id,
        analysis_run.id,
        reclassified_fact.id,
        FactReviewUpdate(
            decision=FactReviewStatus.RECLASSIFIED_AS_ASSUMPTION,
        ),
    )

    inferred_event = sorted(analysis_run.timeline_events, key=lambda item: item.id)[1]
    review_service.review_timeline_event(
        incident.public_id,
        analysis_run.id,
        inferred_event.id,
        TimelineReviewUpdate(
            description="Human review: failure onset remains uncertain.",
        ),
    )

    reviewed_hypothesis = sorted(
        analysis_run.hypotheses,
        key=lambda item: item.rank,
    )[0]
    review_service.review_hypothesis(
        incident.public_id,
        analysis_run.id,
        reviewed_hypothesis.id,
        HypothesisReviewUpdate(
            confidence=73,
            status=HypothesisStatus.SUPPORTED,
        ),
    )
    review_service.add_human_note(
        incident.public_id,
        analysis_run.id,
        HumanNoteCreate(
            note="Human note: validate telemetry before changing capacity.",
        ),
    )

    session.refresh(analysis_run)
    audit_envelope = json.loads(analysis_run.raw_response or "")
    original_hypothesis = audit_envelope["stages"]["hypotheses"]["parsed_output"][
        "hypotheses"
    ][0]
    original_hypothesis["supporting_evidence"].append(
        {
            "reference": {
                "evidence_id": "E-001",
                "line_range": "1-2",
                "excerpt": "this excerpt is not in the redacted evidence",
            },
            "relevance": "The model supplied a mismatched supporting excerpt.",
        }
    )
    original_hypothesis["contradicting_evidence"] = [
        {
            "reference": {
                "evidence_id": "E-001",
                "line_range": "1-2",
            },
            "relevance": "The same log does not identify a database failure.",
        },
        {
            "reference": {
                "evidence_id": "E-001",
                "line_range": "99",
            },
            "relevance": "The model cited a line outside the evidence.",
        },
    ]
    audit_envelope["stages"]["hypotheses"]["raw_response"] = RAW_PROVIDER_SECRET
    analysis_run.raw_response = json.dumps(
        audit_envelope,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    reviewed_hypothesis.contradicting_evidence_codes = ["E-001"]
    reviewed_hypothesis.confidence = 50

    incident.evidence_items.append(
        EvidenceItem(
            evidence_code="E-002",
            source_name="later.log",
            evidence_type=EvidenceType.API_RESPONSE,
            original_text=LATER_EVIDENCE_SECRET,
            redacted_text="[REDACTED]",
            checksum="b" * 64,
        )
    )
    session.commit()
    session.refresh(analysis_run)
    assert analysis_run.raw_response is not None

    return ReviewedReportAnalysis(
        public_id=incident.public_id,
        run_id=analysis_run.id,
        accepted_fact_id=accepted_fact.id,
        unsupported_fact_id=unsupported_fact.id,
        reclassified_fact_id=reclassified_fact.id,
        inferred_event_id=inferred_event.id,
        reviewed_hypothesis_id=reviewed_hypothesis.id,
        raw_response=analysis_run.raw_response,
    )


@pytest.fixture
def reviewed_report_analysis(
    database_session_factory: sessionmaker[Session],
) -> ReviewedReportAnalysis:
    with database_session_factory() as session:
        return _create_reviewed_report_analysis(session)


def test_report_input_uses_reviews_and_retains_original_ai_values(
    database_session_factory: sessionmaker[Session],
    reviewed_report_analysis: ReviewedReportAnalysis,
) -> None:
    with database_session_factory() as session:
        report_input = ReportService(session).build_report_input(
            reviewed_report_analysis.public_id,
            reviewed_report_analysis.run_id,
        )

    assert [fact.fact_id for fact in report_input.confirmed_facts] == [
        reviewed_report_analysis.accepted_fact_id
    ]
    assert report_input.confirmed_facts[0].category is ReportFactCategory.CONFIRMED
    assert report_input.confirmed_facts[0].human_status is FactReviewStatus.ACCEPTED
    assert report_input.confirmed_facts[0].ai_original.claim == (
        report_input.confirmed_facts[0].claim
    )

    assert [fact.fact_id for fact in report_input.unconfirmed_facts] == [
        reviewed_report_analysis.unsupported_fact_id
    ]
    unsupported_fact = report_input.unconfirmed_facts[0]
    assert unsupported_fact.category is ReportFactCategory.UNCONFIRMED
    assert unsupported_fact.human_status is FactReviewStatus.REJECTED
    assert unsupported_fact.ai_original.support_status is (
        ClaimSupportStatus.UNSUPPORTED
    )

    assert [assumption.source for assumption in report_input.assumptions] == [
        ReportAssumptionSource.AI_IDENTIFIED,
        ReportAssumptionSource.HUMAN_RECLASSIFIED_FACT,
    ]
    reclassified = report_input.assumptions[1]
    assert reclassified.originating_fact_id == (
        reviewed_report_analysis.reclassified_fact_id
    )
    assert reclassified.ai_original.fact_support_status is (
        ClaimSupportStatus.SUPPORTED
    )

    direct_event, inferred_event = report_input.timeline
    assert direct_event.is_inferred is False
    assert direct_event.has_human_override is False
    assert inferred_event.event_id == reviewed_report_analysis.inferred_event_id
    assert inferred_event.is_inferred is True
    assert inferred_event.has_human_override is True
    assert inferred_event.description == (
        "Human review: failure onset remains uncertain."
    )
    assert inferred_event.ai_original.description == (
        "Checkout failures may have started before the captured log entry."
    )
    assert inferred_event.uncertainty == (
        "Only one captured failure is available, so the start time cannot be "
        "established."
    )

    reviewed_hypothesis = report_input.hypotheses[0]
    assert reviewed_hypothesis.hypothesis_id == (
        reviewed_report_analysis.reviewed_hypothesis_id
    )
    assert reviewed_hypothesis.confidence == 73
    assert reviewed_hypothesis.validated_ai_confidence == 50
    assert reviewed_hypothesis.ai_original.confidence == 60
    assert reviewed_hypothesis.has_human_confidence_override is True
    assert reviewed_hypothesis.human_status is HypothesisStatus.SUPPORTED
    assert reviewed_hypothesis.supporting_evidence[0].reference.evidence_code == (
        "E-001"
    )
    assert reviewed_hypothesis.contradicting_evidence[0].reference.evidence_code == (
        "E-001"
    )
    assert reviewed_hypothesis.missing_evidence
    assert reviewed_hypothesis.validation_test
    assert reviewed_hypothesis.expected_if_true
    assert reviewed_hypothesis.expected_if_false

    assert report_input.actions[0].linked_hypothesis_ranks == (1,)
    assert report_input.reasoning_risks
    assert report_input.open_questions
    assert [note.note for note in report_input.human_notes] == [
        "Human note: validate telemetry before changing capacity."
    ]


def test_report_input_preserves_summary_unknowns_exactly(
    database_session_factory: sessionmaker[Session],
    reviewed_report_analysis: ReviewedReportAnalysis,
) -> None:
    with database_session_factory() as session:
        report_input = ReportService(session).build_report_input(
            reviewed_report_analysis.public_id,
            reviewed_report_analysis.run_id,
        )

    assert report_input.summary.unknowns == ("The root cause is not verified.",)


def test_report_input_excludes_invalid_existing_code_hypothesis_references(
    database_session_factory: sessionmaker[Session],
    reviewed_report_analysis: ReviewedReportAnalysis,
) -> None:
    with database_session_factory() as session:
        report_input = ReportService(session).build_report_input(
            reviewed_report_analysis.public_id,
            reviewed_report_analysis.run_id,
        )

    hypothesis = report_input.hypotheses[0]
    assert [
        (item.reference.evidence_code, item.line_range)
        for item in hypothesis.supporting_evidence
    ] == [("E-001", "1-2")]
    assert [
        (item.reference.evidence_code, item.line_range)
        for item in hypothesis.contradicting_evidence
    ] == [("E-001", "1-2")]

    assert [
        (
            item.evidence_code,
            item.line_range,
            item.validation_status,
        )
        for item in hypothesis.ai_original.supporting_evidence
    ] == [
        ("E-001", "1-2", EvidenceReferenceValidationStatus.VALID),
        ("E-001", "1-2", EvidenceReferenceValidationStatus.EXCERPT_MISMATCH),
    ]
    assert [
        (
            item.evidence_code,
            item.line_range,
            item.validation_status,
        )
        for item in hypothesis.ai_original.contradicting_evidence
    ] == [
        ("E-001", "1-2", EvidenceReferenceValidationStatus.VALID),
        ("E-001", "99", EvidenceReferenceValidationStatus.INVALID_LINE_RANGE),
    ]


def test_report_input_scopes_evidence_and_excludes_sensitive_sources(
    database_session_factory: sessionmaker[Session],
    reviewed_report_analysis: ReviewedReportAnalysis,
) -> None:
    with database_session_factory() as session:
        report_input = ReportService(session).build_report_input(
            reviewed_report_analysis.public_id,
            reviewed_report_analysis.run_id,
        )
        analysis_run = session.get(AnalysisRun, reviewed_report_analysis.run_id)
        assert analysis_run is not None
        assert analysis_run.raw_response == reviewed_report_analysis.raw_response

    assert [reference.evidence_code for reference in report_input.evidence] == ["E-001"]
    action_references = {
        reference.evidence_code: reference
        for reference in report_input.actions[0].evidence
    }
    assert action_references["E-001"].available is True
    assert action_references["E-002"].available is False
    assert action_references["E-002"].source_name is None
    unsupported_reference = report_input.unconfirmed_facts[0].evidence[0]
    assert unsupported_reference.evidence_code == "E-999"
    assert unsupported_reference.available is False
    assert report_input.validation.unavailable_evidence_codes == ("E-002", "E-999")
    assert report_input.validation.unsupported_fact_ids == (
        reviewed_report_analysis.unsupported_fact_id,
    )

    serialized_input = report_input.model_dump_json()
    assert ORIGINAL_EVIDENCE_SECRET not in serialized_input
    assert LATER_EVIDENCE_SECRET not in serialized_input
    assert RAW_PROVIDER_SECRET not in serialized_input
    assert "raw_response" not in serialized_input
    assert "original_text" not in serialized_input
    assert "redacted_text" not in serialized_input


def test_report_input_rejects_cross_incident_and_noncompleted_runs(
    database_session_factory: sessionmaker[Session],
    reviewed_report_analysis: ReviewedReportAnalysis,
) -> None:
    with database_session_factory() as session:
        other_incident = Incident(
            public_id="INC-000002",
            name="Other incident",
            description="Scope boundary.",
            affected_service="other",
            status=IncidentStatus.ANALYZING,
        )
        running_run = AnalysisRun(
            incident=other_incident,
            provider_name="fake",
            model_name="fixture-v1",
            status=AnalysisRunStatus.RUNNING,
        )
        session.add(other_incident)
        session.commit()

        report_service = ReportService(session)
        with pytest.raises(AnalysisRunNotFoundError):
            report_service.build_report_input(
                other_incident.public_id,
                reviewed_report_analysis.run_id,
            )
        with pytest.raises(
            ReportInputUnavailableError,
            match="must be completed",
        ):
            report_service.build_report_input(
                other_incident.public_id,
                running_run.id,
            )


def test_report_input_requires_validated_stage_data_without_echoing_raw_content(
    database_session_factory: sessionmaker[Session],
) -> None:
    with database_session_factory() as session:
        incident = Incident(
            public_id="INC-000003",
            name="Malformed audit incident",
            description="Safe failure fixture.",
            affected_service="checkout",
            status=IncidentStatus.COMPLETED,
        )
        analysis_run = AnalysisRun(
            incident=incident,
            provider_name="fake",
            model_name="fixture-v1",
            status=AnalysisRunStatus.COMPLETED,
            raw_response=f'{{"secret":"{RAW_PROVIDER_SECRET}"}}',
        )
        session.add(incident)
        session.commit()

        with pytest.raises(
            ReportInputUnavailableError,
            match="missing validated structured report data",
        ) as error:
            ReportService(session).build_report_input(
                incident.public_id,
                analysis_run.id,
            )

    assert RAW_PROVIDER_SECRET not in str(error.value)


def test_generate_draft_persists_all_sections_and_separates_editable_content(
    database_session_factory: sessionmaker[Session],
    reviewed_report_analysis: ReviewedReportAnalysis,
) -> None:
    provider = _RecordingPostmortemProvider(_postmortem_output())

    with database_session_factory() as session:
        report = ReportService(
            session,
            ai_provider=provider,
        ).generate_draft_report(
            reviewed_report_analysis.public_id,
            reviewed_report_analysis.run_id,
        )
        report_id = report.id

    expected_headings = (
        "Executive summary",
        "Incident impact",
        "Detection",
        "Evidence reviewed",
        "Timeline",
        "Confirmed facts",
        "Assumptions and unresolved questions",
        "Root-cause hypotheses and confidence",
        "Supporting and contradicting evidence",
        "Investigation actions",
        "Mitigation and recovery",
        "Biases and reasoning risks",
        "AI limitations and unsupported claims detected",
        "Lessons learned",
        "Follow-up actions",
    )
    assert len(POSTMORTEM_SECTIONS) == len(expected_headings)

    with database_session_factory() as session:
        persisted_report = session.get(Report, report_id)
        assert persisted_report is not None
        assert persisted_report.generated_text == persisted_report.editable_text
        assert persisted_report.final_text is None
        for position, heading in enumerate(expected_headings, start=1):
            assert f"## {position}. {heading}" in persisted_report.generated_text
        assert GENERATED_OUTPUT_SECRET not in persisted_report.generated_text
        assert "[REDACTED_API_KEY]" in persisted_report.generated_text
        assert "<script>" not in persisted_report.generated_text
        assert "&lt;script&gt;" in persisted_report.generated_text

        generation = persisted_report.export_metadata["generation"]
        assert generation["provider_name"] == "fake"
        assert generation["model_name"] == "fixture-v1"
        assert generation["task_prompt"] == {
            "name": "postmortem",
            "version": "v2",
        }
        assert generation["output_schema"] == "postmortem_v1"
        assert len(generation["report_input_sha256"]) == 64
        assert RAW_PROVIDER_SECRET not in json.dumps(generation)


def test_generation_boundary_receives_only_reviewed_secret_safe_report_input(
    database_session_factory: sessionmaker[Session],
    reviewed_report_analysis: ReviewedReportAnalysis,
) -> None:
    provider = _RecordingPostmortemProvider(_postmortem_output())

    with database_session_factory() as session:
        ReportService(session, ai_provider=provider).generate_draft_report(
            reviewed_report_analysis.public_id,
            reviewed_report_analysis.run_id,
        )

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.evidence_manifest is None
    assert request.report_input is not None
    assert request.report_input.timeline[1].description == (
        "Human review: failure onset remains uncertain."
    )
    assert request.report_input.summary.uncertainty == (
        "The frequency and root cause are not yet verified."
    )
    assert request.report_input.summary.unknowns == ("The root cause is not verified.",)
    assert request.report_input.hypotheses[0].confidence == 73
    assert request.report_input.unconfirmed_facts[0].ai_original.support_status is (
        ClaimSupportStatus.UNSUPPORTED
    )

    task_prompt = PromptRegistry().resolve_content(request.prompts.task)
    provider_payload = json.loads(
        GeminiAIProvider._build_contents(request, task_prompt)
    )
    assert "report_input" in provider_payload
    assert "evidence_manifest" not in provider_payload
    serialized_report_input = json.dumps(
        provider_payload["report_input"],
        sort_keys=True,
    )
    assert ORIGINAL_EVIDENCE_SECRET not in serialized_report_input
    assert LATER_EVIDENCE_SECRET not in serialized_report_input
    assert RAW_PROVIDER_SECRET not in serialized_report_input
    assert "raw_response" not in serialized_report_input
    assert "original_text" not in serialized_report_input
    assert "redacted_text" not in serialized_report_input


def test_generate_draft_is_idempotent_for_one_analysis_run(
    database_session_factory: sessionmaker[Session],
    reviewed_report_analysis: ReviewedReportAnalysis,
) -> None:
    provider = _RecordingPostmortemProvider(_postmortem_output())

    with database_session_factory() as session:
        service = ReportService(session, ai_provider=provider)
        first_report = service.generate_draft_report(
            reviewed_report_analysis.public_id,
            reviewed_report_analysis.run_id,
        )
        second_report = service.generate_draft_report(
            reviewed_report_analysis.public_id,
            reviewed_report_analysis.run_id,
        )
        report_count = session.scalar(select(func.count(Report.id)))

    assert second_report.id == first_report.id
    assert report_count == 1
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "failure_category",
    [
        AIFailureCategory.SCHEMA_VALIDATION,
        AIFailureCategory.TRANSIENT_PROVIDER_FAILURE,
    ],
    ids=["invalid-structured-output", "provider-failure"],
)
def test_generation_failure_rolls_back_without_partial_report(
    database_session_factory: sessionmaker[Session],
    reviewed_report_analysis: ReviewedReportAnalysis,
    failure_category: AIFailureCategory,
) -> None:
    provider = _FailingPostmortemProvider(failure_category)

    with database_session_factory() as session:
        with pytest.raises(ReportProviderExecutionError) as error:
            ReportService(session, ai_provider=provider).generate_draft_report(
                reviewed_report_analysis.public_id,
                reviewed_report_analysis.run_id,
            )
        assert session.scalar(select(func.count(Report.id))) == 0

    assert RAW_PROVIDER_SECRET not in str(error.value)
    assert RAW_PROVIDER_SECRET not in repr(error.value)


def test_generation_rejects_mismatched_typed_provider_result(
    database_session_factory: sessionmaker[Session],
    reviewed_report_analysis: ReviewedReportAnalysis,
) -> None:
    provider = _MismatchedPostmortemProvider(_postmortem_output())

    with database_session_factory() as session:
        with pytest.raises(ReportGenerationError, match="invalid postmortem"):
            ReportService(session, ai_provider=provider).generate_draft_report(
                reviewed_report_analysis.public_id,
                reviewed_report_analysis.run_id,
            )
        assert session.scalar(select(func.count(Report.id))) == 0


def test_generation_persistence_error_rolls_back_without_partial_report(
    database_session_factory: sessionmaker[Session],
    reviewed_report_analysis: ReviewedReportAnalysis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _RecordingPostmortemProvider(_postmortem_output())

    with database_session_factory() as session:

        def fail_commit() -> None:
            raise SQLAlchemyError("private database failure")

        monkeypatch.setattr(session, "commit", fail_commit)
        with pytest.raises(ReportPersistenceError, match="could not be saved") as error:
            ReportService(session, ai_provider=provider).generate_draft_report(
                reviewed_report_analysis.public_id,
                reviewed_report_analysis.run_id,
            )

    with database_session_factory() as session:
        assert session.scalar(select(func.count(Report.id))) == 0
    assert "private database failure" not in str(error.value)


def test_generation_rejects_cross_incident_noncompleted_and_incomplete_runs(
    database_session_factory: sessionmaker[Session],
    reviewed_report_analysis: ReviewedReportAnalysis,
) -> None:
    provider = _RecordingPostmortemProvider(_postmortem_output())

    with database_session_factory() as session:
        running_incident = Incident(
            public_id="INC-000002",
            name="Running report incident",
            description="Not completed.",
            affected_service="checkout",
            status=IncidentStatus.ANALYZING,
        )
        running_run = AnalysisRun(
            incident=running_incident,
            provider_name="fake",
            model_name="fixture-v1",
            status=AnalysisRunStatus.RUNNING,
        )
        incomplete_incident = Incident(
            public_id="INC-000003",
            name="Incomplete report incident",
            description="Completed without structured stages.",
            affected_service="checkout",
            status=IncidentStatus.COMPLETED,
        )
        incomplete_run = AnalysisRun(
            incident=incomplete_incident,
            provider_name="fake",
            model_name="fixture-v1",
            status=AnalysisRunStatus.COMPLETED,
            raw_response=f'{{"private":"{RAW_PROVIDER_SECRET}"}}',
        )
        session.add_all([running_incident, incomplete_incident])
        session.commit()

        service = ReportService(session, ai_provider=provider)
        with pytest.raises(AnalysisRunNotFoundError):
            service.generate_draft_report(
                running_incident.public_id,
                reviewed_report_analysis.run_id,
            )
        with pytest.raises(ReportInputUnavailableError, match="must be completed"):
            service.generate_draft_report(
                running_incident.public_id,
                running_run.id,
            )
        with pytest.raises(
            ReportInputUnavailableError,
            match="missing validated structured report data",
        ):
            service.generate_draft_report(
                incomplete_incident.public_id,
                incomplete_run.id,
            )

    assert provider.requests == []


def test_postmortem_output_contract_requires_every_step_13_section() -> None:
    output_data = _postmortem_output().model_dump()
    del output_data["follow_up_actions"]

    with pytest.raises(ValidationError):
        PostmortemOutputV1.model_validate(output_data)
