"""Focused regressions for reviewed, secret-safe report input."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
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
)
from app.schemas.report import (
    ReportAssumptionSource,
    ReportFactCategory,
)
from app.schemas.ai_provider import EvidenceReferenceValidationStatus
from app.schemas.review import (
    FactReviewUpdate,
    HumanNoteCreate,
    HypothesisReviewUpdate,
    TimelineReviewUpdate,
)
from app.services.analysis_service import AnalysisRunNotFoundError, AnalysisService
from app.services.prompt_registry import PromptRegistry
from app.services.providers.fake_provider import FakeAIProvider
from app.services.report_service import ReportInputUnavailableError, ReportService
from app.services.review_service import ReviewService


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_ai_responses.json"
ORIGINAL_EVIDENCE_SECRET = "api_key=original-report-secret"
LATER_EVIDENCE_SECRET = "password=later-report-secret"
RAW_PROVIDER_SECRET = "raw-provider-report-secret"


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
