"""Endpoint regressions for incident-scoped human analysis review."""

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    ClaimSupportStatus,
    Fact,
    FactReviewStatus,
    HumanNote,
    Hypothesis,
    HypothesisConfidenceOverride,
    HypothesisStatus,
    Incident,
    IncidentStatus,
    TimelineEvent,
    TimelineEventReview,
    utc_now,
)


RAW_AUDIT_ENVELOPE = (
    '{"internal":"raw-review-secret","spacing": [1, 2],"nested":{"verbatim":true}}\n'
)


@dataclass(frozen=True, slots=True)
class ReviewableAnalysis:
    public_id: str
    run_id: int
    fact_id: int
    timeline_event_id: int
    hypothesis_id: int


def _create_analysis(
    session_factory: sessionmaker[Session],
    *,
    public_id: str = "INC-000001",
    status: AnalysisRunStatus = AnalysisRunStatus.COMPLETED,
) -> ReviewableAnalysis:
    fact = Fact(
        claim="Checkout requests returned errors.",
        support_status=ClaimSupportStatus.SUPPORTED,
        confidence=92,
        evidence_codes=[],
    )
    timeline_event = TimelineEvent(
        description="AI timeline: checkout errors began.",
        confidence=88,
    )
    hypothesis = Hypothesis(
        rank=1,
        title="Database pool exhaustion",
        explanation="Connections may have been unavailable.",
        confidence=63,
        recommended_test="Inspect pool utilization.",
        expected_true_result="Pool saturation is visible.",
        expected_false_result="Capacity remained available.",
    )
    analysis_run = AnalysisRun(
        model_name="fixture-v1",
        provider_name="fake",
        raw_response=RAW_AUDIT_ENVELOPE,
        status=status,
        completed_at=utc_now() if status is AnalysisRunStatus.COMPLETED else None,
        facts=[fact],
        timeline_events=[timeline_event],
        hypotheses=[hypothesis],
    )
    incident = Incident(
        public_id=public_id,
        name=f"Review fixture {public_id}",
        description="Human review endpoint fixture.",
        affected_service="checkout",
        status=(
            IncidentStatus.COMPLETED
            if status is AnalysisRunStatus.COMPLETED
            else IncidentStatus.ANALYZING
        ),
        analysis_runs=[analysis_run],
    )
    with session_factory() as session:
        session.add(incident)
        session.flush()
        identifiers = ReviewableAnalysis(
            public_id=public_id,
            run_id=analysis_run.id,
            fact_id=fact.id,
            timeline_event_id=timeline_event.id,
            hypothesis_id=hypothesis.id,
        )
        session.commit()
    return identifiers


def _analysis_url(analysis: ReviewableAnalysis) -> str:
    return f"/incidents/{analysis.public_id}/analysis/{analysis.run_id}"


def _assert_ai_values_and_raw_audit_unchanged(
    session_factory: sessionmaker[Session],
    analysis: ReviewableAnalysis,
) -> None:
    with session_factory() as session:
        analysis_run = session.get(AnalysisRun, analysis.run_id)
        fact = session.get(Fact, analysis.fact_id)
        timeline_event = session.get(TimelineEvent, analysis.timeline_event_id)
        hypothesis = session.get(Hypothesis, analysis.hypothesis_id)

        assert analysis_run is not None
        assert analysis_run.raw_response is not None
        assert analysis_run.raw_response.encode("utf-8") == (
            RAW_AUDIT_ENVELOPE.encode("utf-8")
        )
        assert fact is not None
        assert fact.claim == "Checkout requests returned errors."
        assert fact.support_status is ClaimSupportStatus.SUPPORTED
        assert fact.confidence == 92
        assert fact.evidence_codes == []
        assert timeline_event is not None
        assert timeline_event.description == "AI timeline: checkout errors began."
        assert timeline_event.confidence == 88
        assert timeline_event.evidence_codes == []
        assert hypothesis is not None
        assert hypothesis.title == "Database pool exhaustion"
        assert hypothesis.explanation == "Connections may have been unavailable."
        assert hypothesis.confidence == 63
        assert hypothesis.recommended_test == "Inspect pool utilization."
        assert hypothesis.expected_true_result == "Pool saturation is visible."
        assert hypothesis.expected_false_result == "Capacity remained available."


def test_analysis_page_exposes_human_review_controls_without_raw_audit(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    analysis = _create_analysis(database_session_factory)

    response = database_client.get(_analysis_url(analysis))

    assert response.status_code == 200
    assert "Human review notes" in response.text
    assert "Add a human note" in response.text
    assert "Human decision" in response.text
    assert "Reclassify as assumption" in response.text
    assert "Human-reviewed description" in response.text
    assert "Human hypothesis decision" in response.text
    assert "Confirmed By Human" in response.text
    assert (
        f"/analysis/{analysis.run_id}/facts/{analysis.fact_id}/review" in response.text
    )
    assert (
        f"/analysis/{analysis.run_id}/timeline/"
        f"{analysis.timeline_event_id}/review" in response.text
    )
    assert (
        f"/analysis/{analysis.run_id}/hypotheses/"
        f"{analysis.hypothesis_id}/review" in response.text
    )
    assert "raw-review-secret" not in response.text
    assert '"raw_response"' not in response.text
    assert 'data-hypothesis-review-status="UNTESTED"' in response.text
    assert 'data-hypothesis-review-status="CONFIRMED_BY_HUMAN"' not in (response.text)
    _assert_ai_values_and_raw_audit_unchanged(
        database_session_factory,
        analysis,
    )


@pytest.mark.parametrize(
    ("decision", "display_label"),
    [
        (FactReviewStatus.ACCEPTED, "Accepted"),
        (FactReviewStatus.REJECTED, "Rejected"),
        (
            FactReviewStatus.RECLASSIFIED_AS_ASSUMPTION,
            "Reclassified As Assumption",
        ),
    ],
)
def test_fact_review_decisions_persist_and_are_visible(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
    decision: FactReviewStatus,
    display_label: str,
) -> None:
    analysis = _create_analysis(database_session_factory)

    response = database_client.post(
        f"{_analysis_url(analysis)}/facts/{analysis.fact_id}/review",
        data={"decision": decision.value},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == (
        f"{_analysis_url(analysis)}?notice=analysis-review-updated"
        "#facts-assumptions-section"
    )
    _assert_ai_values_and_raw_audit_unchanged(
        database_session_factory,
        analysis,
    )
    with database_session_factory() as session:
        fact = session.get(Fact, analysis.fact_id)
        assert fact is not None
        assert fact.claim == "Checkout requests returned errors."
        assert fact.support_status is ClaimSupportStatus.SUPPORTED
        assert fact.human_status is decision

    detail_response = database_client.get(_analysis_url(analysis))
    assert detail_response.status_code == 200
    assert display_label in detail_response.text
    if decision is FactReviewStatus.RECLASSIFIED_AS_ASSUMPTION:
        assert "Human-reviewed fact reclassified as assumption" in (
            detail_response.text
        )
        assert "Original AI classification: SUPPORTED" in detail_response.text
    assert "raw-review-secret" not in detail_response.text


def test_timeline_human_edit_preserves_and_displays_the_ai_description(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    analysis = _create_analysis(database_session_factory)
    reviewed_description = "Human review: errors started after the alert."

    response = database_client.post(
        (f"{_analysis_url(analysis)}/timeline/{analysis.timeline_event_id}/review"),
        data={"description": reviewed_description},
        follow_redirects=False,
    )

    assert response.status_code == 303
    _assert_ai_values_and_raw_audit_unchanged(
        database_session_factory,
        analysis,
    )
    with database_session_factory() as session:
        event = session.get(TimelineEvent, analysis.timeline_event_id)
        review = session.scalar(
            select(TimelineEventReview).where(
                TimelineEventReview.timeline_event_id == analysis.timeline_event_id
            )
        )
        assert event is not None
        assert event.description == "AI timeline: checkout errors began."
        assert review is not None
        assert review.description == reviewed_description

    detail_response = database_client.get(_analysis_url(analysis))
    assert reviewed_description in detail_response.text
    assert "Human edit" in detail_response.text
    assert "Original AI description retained" in detail_response.text
    assert "AI timeline: checkout errors began." in detail_response.text
    assert "raw-review-secret" not in detail_response.text


def test_human_note_is_persisted_and_escaped(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    analysis = _create_analysis(database_session_factory)
    note = "<script>alert('review')</script>"

    response = database_client.post(
        f"{_analysis_url(analysis)}/notes",
        data={"note": note},
        follow_redirects=False,
    )

    assert response.status_code == 303
    _assert_ai_values_and_raw_audit_unchanged(
        database_session_factory,
        analysis,
    )
    with database_session_factory() as session:
        notes = session.scalars(select(HumanNote)).all()
        assert [item.note for item in notes] == [note]

    detail_response = database_client.get(_analysis_url(analysis))
    assert "&lt;script&gt;" in detail_response.text
    assert "<script>alert" not in detail_response.text
    assert "Human note added" in detail_response.text
    assert "raw-review-secret" not in detail_response.text


@pytest.mark.parametrize("hypothesis_status", list(HypothesisStatus))
def test_every_hypothesis_status_and_confidence_override_persists_visibly(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
    hypothesis_status: HypothesisStatus,
) -> None:
    analysis = _create_analysis(database_session_factory)

    response = database_client.post(
        (f"{_analysis_url(analysis)}/hypotheses/{analysis.hypothesis_id}/review"),
        data={
            "confidence": "41",
            "hypothesis_status": hypothesis_status.value,
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    _assert_ai_values_and_raw_audit_unchanged(
        database_session_factory,
        analysis,
    )
    with database_session_factory() as session:
        hypothesis = session.get(Hypothesis, analysis.hypothesis_id)
        confidence_override = session.scalar(
            select(HypothesisConfidenceOverride).where(
                HypothesisConfidenceOverride.hypothesis_id == analysis.hypothesis_id
            )
        )
        assert hypothesis is not None
        assert hypothesis.confidence == 63
        assert hypothesis.status is hypothesis_status
        assert confidence_override is not None
        assert confidence_override.confidence == 41

    detail_response = database_client.get(_analysis_url(analysis))
    assert "Confidence" in detail_response.text
    assert "41%" in detail_response.text
    assert "Human override" in detail_response.text
    assert "Original AI confidence retained: 63%." in detail_response.text
    assert f"Review status {hypothesis_status.value}" in detail_response.text
    assert "raw-review-secret" not in detail_response.text


def test_repeated_review_updates_preserve_ai_values_and_reopen_with_latest_values(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    analysis = _create_analysis(database_session_factory)
    fact_url = f"{_analysis_url(analysis)}/facts/{analysis.fact_id}/review"
    timeline_url = (
        f"{_analysis_url(analysis)}/timeline/{analysis.timeline_event_id}/review"
    )
    hypothesis_url = (
        f"{_analysis_url(analysis)}/hypotheses/{analysis.hypothesis_id}/review"
    )

    review_submissions = [
        (fact_url, {"decision": "ACCEPTED"}),
        (fact_url, {"decision": "REJECTED"}),
        (fact_url, {"decision": "RECLASSIFIED_AS_ASSUMPTION"}),
        (timeline_url, {"description": "First human timeline edit."}),
        (timeline_url, {"description": "Latest human timeline edit."}),
        (f"{_analysis_url(analysis)}/notes", {"note": "First human note."}),
        (f"{_analysis_url(analysis)}/notes", {"note": "Second human note."}),
        (
            hypothesis_url,
            {"confidence": "54", "hypothesis_status": "SUPPORTED"},
        ),
        (
            hypothesis_url,
            {"confidence": "37", "hypothesis_status": "WEAKENED"},
        ),
    ]
    for url, form_data in review_submissions:
        response = database_client.post(
            url,
            data=form_data,
            follow_redirects=False,
        )
        assert response.status_code == 303
        _assert_ai_values_and_raw_audit_unchanged(
            database_session_factory,
            analysis,
        )

    with database_session_factory() as session:
        fact = session.get(Fact, analysis.fact_id)
        timeline_reviews = session.scalars(select(TimelineEventReview)).all()
        human_notes = session.scalars(select(HumanNote).order_by(HumanNote.id)).all()
        hypothesis = session.get(Hypothesis, analysis.hypothesis_id)
        confidence_overrides = session.scalars(
            select(HypothesisConfidenceOverride)
        ).all()
        assert fact is not None
        assert fact.human_status is FactReviewStatus.RECLASSIFIED_AS_ASSUMPTION
        assert [item.description for item in timeline_reviews] == [
            "Latest human timeline edit."
        ]
        assert [item.note for item in human_notes] == [
            "First human note.",
            "Second human note.",
        ]
        assert hypothesis is not None
        assert hypothesis.status is HypothesisStatus.WEAKENED
        assert [item.confidence for item in confidence_overrides] == [37]

    reopened_response = database_client.get(_analysis_url(analysis))
    assert reopened_response.status_code == 200
    assert "Human-reviewed fact reclassified as assumption" in (reopened_response.text)
    assert "Latest human timeline edit." in reopened_response.text
    assert "First human timeline edit." not in reopened_response.text
    assert "Original AI description retained" in reopened_response.text
    assert "First human note." in reopened_response.text
    assert "Second human note." in reopened_response.text
    assert "Confidence 37%" in reopened_response.text
    assert "Original AI confidence retained: 63%." in reopened_response.text
    assert 'data-hypothesis-review-status="WEAKENED"' in reopened_response.text
    assert "raw-review-secret" not in reopened_response.text
    assert '"raw_response"' not in reopened_response.text


def test_confirmed_by_human_requires_explicit_scoped_submission(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    first = _create_analysis(database_session_factory, public_id="INC-000001")
    second = _create_analysis(database_session_factory, public_id="INC-000002")

    initial_response = database_client.get(_analysis_url(first))
    assert 'data-hypothesis-review-status="UNTESTED"' in initial_response.text
    assert 'data-hypothesis-review-status="CONFIRMED_BY_HUMAN"' not in (
        initial_response.text
    )

    cross_run_response = database_client.post(
        (f"{_analysis_url(first)}/hypotheses/{second.hypothesis_id}/review"),
        data={
            "confidence": "90",
            "hypothesis_status": "CONFIRMED_BY_HUMAN",
        },
    )
    assert cross_run_response.status_code == 404
    _assert_ai_values_and_raw_audit_unchanged(
        database_session_factory,
        first,
    )
    _assert_ai_values_and_raw_audit_unchanged(
        database_session_factory,
        second,
    )
    with database_session_factory() as session:
        first_hypothesis = session.get(Hypothesis, first.hypothesis_id)
        second_hypothesis = session.get(Hypothesis, second.hypothesis_id)
        assert first_hypothesis is not None
        assert first_hypothesis.status is HypothesisStatus.UNTESTED
        assert second_hypothesis is not None
        assert second_hypothesis.status is HypothesisStatus.UNTESTED

    explicit_response = database_client.post(
        (f"{_analysis_url(first)}/hypotheses/{first.hypothesis_id}/review"),
        data={
            "confidence": "90",
            "hypothesis_status": "CONFIRMED_BY_HUMAN",
        },
        follow_redirects=False,
    )
    assert explicit_response.status_code == 303
    _assert_ai_values_and_raw_audit_unchanged(
        database_session_factory,
        first,
    )

    reopened_response = database_client.get(_analysis_url(first))
    assert (
        'data-hypothesis-review-status="CONFIRMED_BY_HUMAN"' in reopened_response.text
    )
    assert "Human override" in reopened_response.text
    assert "Original AI confidence retained: 63%." in reopened_response.text
    assert "raw-review-secret" not in reopened_response.text
    assert '"raw_response"' not in reopened_response.text


def test_invalid_review_values_and_object_ids_are_rejected_safely(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    analysis = _create_analysis(database_session_factory)

    pending_response = database_client.post(
        f"{_analysis_url(analysis)}/facts/{analysis.fact_id}/review",
        data={"decision": FactReviewStatus.PENDING.value},
    )
    invalid_status_response = database_client.post(
        (f"{_analysis_url(analysis)}/hypotheses/{analysis.hypothesis_id}/review"),
        data={"confidence": "40", "hypothesis_status": "PROVEN"},
    )
    invalid_confidence_response = database_client.post(
        (f"{_analysis_url(analysis)}/hypotheses/{analysis.hypothesis_id}/review"),
        data={"confidence": "101", "hypothesis_status": "SUPPORTED"},
    )
    missing_target_response = database_client.post(
        f"{_analysis_url(analysis)}/facts/999999/review",
        data={"decision": FactReviewStatus.ACCEPTED.value},
    )
    blank_timeline_response = database_client.post(
        (f"{_analysis_url(analysis)}/timeline/{analysis.timeline_event_id}/review"),
        data={"description": "   "},
    )
    blank_note_response = database_client.post(
        f"{_analysis_url(analysis)}/notes",
        data={"note": "   "},
    )

    assert pending_response.status_code == 422
    assert invalid_status_response.status_code == 422
    assert invalid_confidence_response.status_code == 422
    assert missing_target_response.status_code == 404
    assert blank_timeline_response.status_code == 422
    assert blank_note_response.status_code == 422
    for response in (
        pending_response,
        invalid_status_response,
        invalid_confidence_response,
        missing_target_response,
        blank_timeline_response,
        blank_note_response,
    ):
        assert "raw-review-secret" not in response.text
    _assert_ai_values_and_raw_audit_unchanged(
        database_session_factory,
        analysis,
    )
    with database_session_factory() as session:
        assert session.scalar(select(TimelineEventReview)) is None
        assert session.scalar(select(HumanNote)) is None


def test_cross_run_and_cross_incident_review_attempts_are_not_found(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    first = _create_analysis(database_session_factory, public_id="INC-000001")
    second = _create_analysis(database_session_factory, public_id="INC-000002")

    responses = [
        database_client.post(
            f"{_analysis_url(first)}/facts/{second.fact_id}/review",
            data={"decision": FactReviewStatus.ACCEPTED.value},
        ),
        database_client.post(
            f"{_analysis_url(first)}/timeline/{second.timeline_event_id}/review",
            data={"description": "Cross-run edit"},
        ),
        database_client.post(
            f"{_analysis_url(first)}/hypotheses/{second.hypothesis_id}/review",
            data={"confidence": "50", "hypothesis_status": "SUPPORTED"},
        ),
        database_client.post(
            f"/incidents/{first.public_id}/analysis/{second.run_id}/notes",
            data={"note": "Cross-incident note"},
        ),
    ]

    assert [response.status_code for response in responses] == [404, 404, 404, 404]
    _assert_ai_values_and_raw_audit_unchanged(
        database_session_factory,
        first,
    )
    _assert_ai_values_and_raw_audit_unchanged(
        database_session_factory,
        second,
    )
    with database_session_factory() as session:
        second_fact = session.get(Fact, second.fact_id)
        second_timeline_review = session.scalar(
            select(TimelineEventReview).where(
                TimelineEventReview.timeline_event_id == second.timeline_event_id
            )
        )
        second_hypothesis = session.get(Hypothesis, second.hypothesis_id)
        assert second_fact is not None
        assert second_fact.human_status is FactReviewStatus.PENDING
        assert second_timeline_review is None
        assert second_hypothesis is not None
        assert second_hypothesis.status is HypothesisStatus.UNTESTED
        assert session.scalar(select(HumanNote)) is None


def test_non_completed_analysis_rejects_review_transition(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    analysis = _create_analysis(
        database_session_factory,
        status=AnalysisRunStatus.RUNNING,
    )

    response = database_client.post(
        f"{_analysis_url(analysis)}/facts/{analysis.fact_id}/review",
        data={"decision": FactReviewStatus.ACCEPTED.value},
    )

    assert response.status_code == 409
    assert "must be completed before human review" in response.text
    assert "raw-review-secret" not in response.text
    _assert_ai_values_and_raw_audit_unchanged(
        database_session_factory,
        analysis,
    )
    with database_session_factory() as session:
        fact = session.get(Fact, analysis.fact_id)
        assert fact is not None
        assert fact.human_status is FactReviewStatus.PENDING
