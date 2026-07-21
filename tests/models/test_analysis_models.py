"""Tests for analysis-run and structured reasoning persistence models."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import CheckConstraint, UniqueConstraint, delete, func, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    BiasFlag,
    ClaimSupportStatus,
    Fact,
    FactReviewStatus,
    Hypothesis,
    HypothesisStatus,
    Incident,
    RecommendedAction,
    TimelineEvent,
)
from app.models.analysis import recommended_action_hypotheses


def test_analysis_review_enum_values_are_locked() -> None:
    assert [status.value for status in AnalysisRunStatus] == [
        "RUNNING",
        "COMPLETED",
        "FAILED",
    ]
    assert [status.value for status in FactReviewStatus] == [
        "PENDING",
        "ACCEPTED",
        "REJECTED",
        "RECLASSIFIED_AS_ASSUMPTION",
    ]
    assert [status.value for status in HypothesisStatus] == [
        "UNTESTED",
        "SUPPORTED",
        "WEAKENED",
        "REJECTED",
        "CONFIRMED_BY_HUMAN",
    ]


def test_analysis_models_expose_locked_fields_and_relationships() -> None:
    expected_columns = {
        AnalysisRun: {
            "id",
            "incident_id",
            "model_name",
            "provider_name",
            "prompt_versions",
            "input_evidence_codes",
            "raw_response",
            "error_message",
            "status",
            "started_at",
            "completed_at",
        },
        Fact: {
            "id",
            "analysis_run_id",
            "claim",
            "support_status",
            "confidence",
            "evidence_codes",
            "supporting_excerpt",
            "human_status",
        },
        TimelineEvent: {
            "id",
            "analysis_run_id",
            "event_time",
            "description",
            "evidence_codes",
            "is_inferred",
            "confidence",
        },
        Hypothesis: {
            "id",
            "analysis_run_id",
            "rank",
            "title",
            "explanation",
            "confidence",
            "supporting_evidence_codes",
            "contradicting_evidence_codes",
            "missing_evidence",
            "recommended_test",
            "expected_true_result",
            "expected_false_result",
            "status",
        },
        BiasFlag: {
            "id",
            "analysis_run_id",
            "bias_type",
            "explanation",
            "trigger",
            "mitigation",
            "confidence",
        },
        RecommendedAction: {
            "id",
            "analysis_run_id",
            "description",
            "priority",
            "evidence_codes",
            "owner_role",
            "expected_information",
            "operational_risk",
        },
    }

    for model, column_names in expected_columns.items():
        assert set(inspect(model).columns.keys()) == column_names

    assert set(inspect(AnalysisRun).relationships.keys()) == {
        "actions",
        "bias_flags",
        "facts",
        "hypotheses",
        "incident",
        "reports",
        "timeline_events",
    }
    for model in (Fact, TimelineEvent, BiasFlag):
        assert set(inspect(model).relationships.keys()) == {"analysis_run"}
    assert set(inspect(Hypothesis).relationships.keys()) == {
        "analysis_run",
        "recommended_actions",
    }
    assert set(inspect(RecommendedAction).relationships.keys()) == {
        "analysis_run",
        "hypotheses",
    }

    assert AnalysisRun.__table__.c.status.type.enum_class is AnalysisRunStatus
    assert Fact.__table__.c.support_status.type.enum_class is ClaimSupportStatus
    assert Fact.__table__.c.human_status.type.enum_class is FactReviewStatus
    assert Hypothesis.__table__.c.status.type.enum_class is HypothesisStatus
    assert AnalysisRun.__table__.c.raw_response.nullable is True
    assert AnalysisRun.__table__.c.error_message.nullable is True
    assert AnalysisRun.__table__.c.completed_at.nullable is True
    assert set(recommended_action_hypotheses.c.keys()) == {
        "hypothesis_id",
        "recommended_action_id",
    }
    assert all(
        foreign_key.ondelete == "CASCADE"
        for foreign_key in recommended_action_hypotheses.foreign_keys
    )

    fact_constraint_names = {
        constraint.name
        for constraint in Fact.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }
    hypothesis_constraint_names = {
        constraint.name for constraint in Hypothesis.__table__.constraints
    }
    assert "ck_facts_confidence_range" in fact_constraint_names
    assert "ck_hypotheses_positive_rank" in hypothesis_constraint_names
    assert "ck_hypotheses_confidence_range" in hypothesis_constraint_names
    assert any(
        isinstance(constraint, UniqueConstraint)
        and tuple(column.name for column in constraint.columns)
        == ("analysis_run_id", "rank")
        for constraint in Hypothesis.__table__.constraints
    )


def test_complete_analysis_run_persists_structured_results_and_audit_data(
    model_session_factory: sessionmaker[Session],
) -> None:
    event_time = datetime(2025, 1, 1, 10, 5, tzinfo=UTC)
    completed_at = datetime(2025, 1, 1, 10, 10, tzinfo=UTC)

    with model_session_factory() as session:
        hypothesis = Hypothesis(
            rank=1,
            title="Database pool exhaustion",
            explanation="Connections were unavailable",
            confidence=70,
            supporting_evidence_codes=["E-001"],
            contradicting_evidence_codes=["E-002"],
            missing_evidence=["pool configuration"],
            recommended_test="Compare pool settings",
            expected_true_result="Pool limit decreased",
            expected_false_result="Pool settings unchanged",
        )
        analysis_run = AnalysisRun(
            model_name="fake-analysis-v1",
            provider_name="fake",
            prompt_versions={"summary": "summary_v1", "timeline": "timeline_v1"},
            input_evidence_codes=["E-001", "E-002"],
            raw_response='{"summary":"Checkout failures observed"}',
            status=AnalysisRunStatus.COMPLETED,
            completed_at=completed_at,
            facts=[
                Fact(
                    claim="Checkout returned errors",
                    support_status=ClaimSupportStatus.SUPPORTED,
                    confidence=95,
                    evidence_codes=["E-001"],
                    supporting_excerpt="checkout failed",
                )
            ],
            timeline_events=[
                TimelineEvent(
                    event_time=event_time,
                    description="Errors began",
                    evidence_codes=["E-001"],
                    confidence=90,
                )
            ],
            hypotheses=[hypothesis],
            bias_flags=[
                BiasFlag(
                    bias_type="ANCHORING",
                    explanation="Deployment timing may dominate reasoning",
                    trigger="Deployment preceded errors",
                    mitigation="Compare non-deployment causes",
                    confidence=60,
                )
            ],
            actions=[
                RecommendedAction(
                    description="Compare database pool settings",
                    priority="HIGH",
                    hypotheses=[hypothesis],
                    evidence_codes=["E-001"],
                    owner_role="Site reliability engineer",
                    expected_information="Whether pool limits changed",
                    operational_risk="Low",
                )
            ],
        )
        incident = Incident(
            public_id="INC-000001",
            name="Checkout failures",
            description="Intermittent checkout errors",
            affected_service="checkout",
            analysis_runs=[analysis_run],
        )
        session.add(incident)
        session.flush()
        analysis_run_id = analysis_run.id
        session.commit()

    with model_session_factory() as session:
        loaded_run = session.get(AnalysisRun, analysis_run_id)
        assert loaded_run is not None

        assert loaded_run.incident.public_id == "INC-000001"
        assert loaded_run.model_name == "fake-analysis-v1"
        assert loaded_run.provider_name == "fake"
        assert loaded_run.prompt_versions == {
            "summary": "summary_v1",
            "timeline": "timeline_v1",
        }
        assert loaded_run.input_evidence_codes == ["E-001", "E-002"]
        assert loaded_run.raw_response == '{"summary":"Checkout failures observed"}'
        assert loaded_run.error_message is None
        assert loaded_run.status is AnalysisRunStatus.COMPLETED
        assert loaded_run.started_at.tzinfo is UTC
        assert loaded_run.completed_at == completed_at

        fact = loaded_run.facts[0]
        assert fact.support_status is ClaimSupportStatus.SUPPORTED
        assert fact.human_status is FactReviewStatus.PENDING
        assert fact.evidence_codes == ["E-001"]

        timeline_event = loaded_run.timeline_events[0]
        assert timeline_event.event_time == event_time
        assert timeline_event.is_inferred is False

        hypothesis = loaded_run.hypotheses[0]
        assert hypothesis.rank == 1
        assert hypothesis.supporting_evidence_codes == ["E-001"]
        assert hypothesis.contradicting_evidence_codes == ["E-002"]
        assert hypothesis.missing_evidence == ["pool configuration"]
        assert hypothesis.status is HypothesisStatus.UNTESTED

        assert loaded_run.bias_flags[0].bias_type == "ANCHORING"
        action = loaded_run.actions[0]
        assert action.hypotheses[0].id == hypothesis.id
        assert action.hypotheses[0].title == "Database pool exhaustion"
        assert action.evidence_codes == ["E-001"]


def test_analysis_defaults_persist_across_structured_results(
    model_session_factory: sessionmaker[Session],
) -> None:
    analysis_run = AnalysisRun(
        model_name="fake-analysis-v1",
        provider_name="fake",
        facts=[
            Fact(
                claim="Checkout returned errors",
                support_status=ClaimSupportStatus.SUPPORTED,
                confidence=90,
            )
        ],
        timeline_events=[
            TimelineEvent(
                description="Errors began",
                confidence=80,
            )
        ],
        hypotheses=[
            Hypothesis(
                rank=1,
                title="Database pool exhaustion",
                explanation="Connections may have been unavailable",
                confidence=70,
                recommended_test="Inspect pool metrics",
                expected_true_result="Pool saturation is visible",
                expected_false_result="Pool capacity remained available",
            )
        ],
        actions=[
            RecommendedAction(
                description="Inspect database pool metrics",
                priority="HIGH",
                owner_role="Site reliability engineer",
                expected_information="Whether the pool was saturated",
                operational_risk="Low",
            )
        ],
    )
    incident = Incident(
        public_id="INC-000001",
        name="Checkout failures",
        description="Intermittent checkout errors",
        affected_service="checkout",
        analysis_runs=[analysis_run],
    )

    with model_session_factory() as session:
        session.add(incident)
        session.flush()
        analysis_run_id = analysis_run.id
        session.commit()

    with model_session_factory() as session:
        loaded_run = session.get(AnalysisRun, analysis_run_id)
        assert loaded_run is not None
        assert loaded_run.prompt_versions == {}
        assert loaded_run.input_evidence_codes == []
        assert loaded_run.raw_response is None
        assert loaded_run.error_message is None
        assert loaded_run.status is AnalysisRunStatus.RUNNING
        assert loaded_run.started_at.tzinfo is UTC
        assert loaded_run.completed_at is None

        fact = loaded_run.facts[0]
        assert fact.evidence_codes == []
        assert fact.supporting_excerpt is None
        assert fact.human_status is FactReviewStatus.PENDING

        timeline_event = loaded_run.timeline_events[0]
        assert timeline_event.event_time is None
        assert timeline_event.evidence_codes == []
        assert timeline_event.is_inferred is False

        hypothesis = loaded_run.hypotheses[0]
        assert hypothesis.supporting_evidence_codes == []
        assert hypothesis.contradicting_evidence_codes == []
        assert hypothesis.missing_evidence == []
        assert hypothesis.status is HypothesisStatus.UNTESTED

        assert loaded_run.actions[0].evidence_codes == []


def test_analysis_confidence_outside_locked_range_is_rejected(
    model_session_factory: sessionmaker[Session],
) -> None:
    incident = Incident(
        public_id="INC-000001",
        name="Checkout failures",
        description="Intermittent checkout errors",
        affected_service="checkout",
        analysis_runs=[
            AnalysisRun(
                model_name="fake-analysis-v1",
                provider_name="fake",
                facts=[
                    Fact(
                        claim="Unsupported certainty",
                        support_status=ClaimSupportStatus.UNSUPPORTED,
                        confidence=101,
                    )
                ],
            )
        ],
    )

    with model_session_factory() as session:
        session.add(incident)
        with pytest.raises(IntegrityError):
            session.commit()


def test_hypothesis_rank_must_be_unique_within_analysis_run(
    model_session_factory: sessionmaker[Session],
) -> None:
    duplicate_rank_hypotheses = [
        Hypothesis(
            rank=1,
            title=title,
            explanation="Possible explanation",
            confidence=60,
            recommended_test="Run a focused check",
            expected_true_result="Supporting signal appears",
            expected_false_result="Supporting signal is absent",
        )
        for title in ("First hypothesis", "Second hypothesis")
    ]
    incident = Incident(
        public_id="INC-000001",
        name="Checkout failures",
        description="Intermittent checkout errors",
        affected_service="checkout",
        analysis_runs=[
            AnalysisRun(
                model_name="fake-analysis-v1",
                provider_name="fake",
                hypotheses=duplicate_rank_hypotheses,
            )
        ],
    )

    with model_session_factory() as session:
        session.add(incident)
        with pytest.raises(IntegrityError):
            session.commit()


def test_in_place_json_mutations_persist(
    model_session_factory: sessionmaker[Session],
) -> None:
    with model_session_factory() as session:
        analysis_run = AnalysisRun(
            model_name="fake-analysis-v1",
            provider_name="fake",
            prompt_versions={"summary": "summary_v1"},
            input_evidence_codes=["E-001"],
        )
        incident = Incident(
            public_id="INC-000001",
            name="Checkout failures",
            description="Intermittent checkout errors",
            affected_service="checkout",
            analysis_runs=[analysis_run],
        )
        session.add(incident)
        session.flush()
        analysis_run_id = analysis_run.id
        session.commit()

    with model_session_factory() as session:
        loaded_run = session.get(AnalysisRun, analysis_run_id)
        assert loaded_run is not None
        loaded_run.prompt_versions["critic"] = "critic_v1"
        loaded_run.input_evidence_codes.append("E-002")
        session.commit()

    with model_session_factory() as session:
        reloaded_run = session.get(AnalysisRun, analysis_run_id)
        assert reloaded_run is not None
        assert reloaded_run.prompt_versions == {
            "summary": "summary_v1",
            "critic": "critic_v1",
        }
        assert reloaded_run.input_evidence_codes == ["E-001", "E-002"]


def test_provider_and_failure_metadata_persist(
    model_session_factory: sessionmaker[Session],
) -> None:
    with model_session_factory() as session:
        analysis_run = AnalysisRun(
            model_name="external-analysis-v1",
            provider_name="openai",
            raw_response='{"summary":"partial result"}',
            status=AnalysisRunStatus.FAILED,
            error_message="Structured response validation failed",
        )
        incident = Incident(
            public_id="INC-000001",
            name="Checkout failures",
            description="Intermittent checkout errors",
            affected_service="checkout",
            analysis_runs=[analysis_run],
        )
        session.add(incident)
        session.flush()
        analysis_run_id = analysis_run.id
        session.commit()

    with model_session_factory() as session:
        loaded_run = session.get(AnalysisRun, analysis_run_id)
        assert loaded_run is not None
        assert loaded_run.provider_name == "openai"
        assert loaded_run.model_name == "external-analysis-v1"
        assert loaded_run.status is AnalysisRunStatus.FAILED
        assert loaded_run.error_message == "Structured response validation failed"
        assert loaded_run.raw_response == '{"summary":"partial result"}'


def test_deleting_hypothesis_cleans_up_action_association(
    sqlite_engine: Engine,
    model_session_factory: sessionmaker[Session],
) -> None:
    with model_session_factory() as session:
        hypothesis = Hypothesis(
            rank=1,
            title="Database pool exhaustion",
            explanation="Connections were unavailable",
            confidence=70,
            recommended_test="Compare pool settings",
            expected_true_result="Pool limit decreased",
            expected_false_result="Pool settings unchanged",
        )
        action = RecommendedAction(
            description="Compare database pool settings",
            priority="HIGH",
            hypotheses=[hypothesis],
            owner_role="Site reliability engineer",
            expected_information="Whether pool limits changed",
            operational_risk="Low",
        )
        analysis_run = AnalysisRun(
            model_name="fake-analysis-v1",
            provider_name="fake",
            hypotheses=[hypothesis],
            actions=[action],
        )
        incident = Incident(
            public_id="INC-000001",
            name="Checkout failures",
            description="Intermittent checkout errors",
            affected_service="checkout",
            analysis_runs=[analysis_run],
        )
        session.add(incident)
        session.flush()
        hypothesis_id = hypothesis.id
        action_id = action.id
        session.commit()

    with sqlite_engine.begin() as connection:
        result = connection.execute(
            delete(Hypothesis).where(Hypothesis.id == hypothesis_id),
        )
        association_count = connection.execute(
            select(func.count()).select_from(recommended_action_hypotheses),
        ).scalar_one()
        assert result.rowcount == 1
        assert association_count == 0

    with model_session_factory() as session:
        loaded_action = session.get(RecommendedAction, action_id)
        assert loaded_action is not None
        assert loaded_action.hypotheses == []


def test_database_level_incident_delete_cascades_through_analysis_results(
    sqlite_engine: Engine,
    model_session_factory: sessionmaker[Session],
) -> None:
    with model_session_factory() as session:
        fact = Fact(
            claim="Checkout returned errors",
            support_status=ClaimSupportStatus.SUPPORTED,
            confidence=95,
        )
        timeline_event = TimelineEvent(
            description="Errors began",
            confidence=90,
        )
        hypothesis = Hypothesis(
            rank=1,
            title="Database pool exhaustion",
            explanation="Connections were unavailable",
            confidence=70,
            recommended_test="Compare pool settings",
            expected_true_result="Pool limit decreased",
            expected_false_result="Pool settings unchanged",
        )
        bias_flag = BiasFlag(
            bias_type="ANCHORING",
            explanation="Deployment timing may dominate reasoning",
            trigger="Deployment preceded errors",
            mitigation="Compare non-deployment causes",
            confidence=60,
        )
        action = RecommendedAction(
            description="Compare database pool settings",
            priority="HIGH",
            hypotheses=[hypothesis],
            owner_role="Site reliability engineer",
            expected_information="Whether pool limits changed",
            operational_risk="Low",
        )
        analysis_run = AnalysisRun(
            model_name="fake-analysis-v1",
            provider_name="fake",
            facts=[fact],
            timeline_events=[timeline_event],
            hypotheses=[hypothesis],
            bias_flags=[bias_flag],
            actions=[action],
        )
        incident = Incident(
            public_id="INC-000001",
            name="Checkout failures",
            description="Intermittent checkout errors",
            affected_service="checkout",
            analysis_runs=[analysis_run],
        )
        session.add(incident)
        session.flush()
        incident_id = incident.id
        analysis_run_id = analysis_run.id
        fact_id = fact.id
        timeline_event_id = timeline_event.id
        hypothesis_id = hypothesis.id
        bias_flag_id = bias_flag.id
        action_id = action.id
        session.commit()

    with sqlite_engine.begin() as connection:
        result = connection.execute(
            delete(Incident).where(Incident.id == incident_id),
        )
        assert result.rowcount == 1

    with model_session_factory() as session:
        assert session.get(Incident, incident_id) is None
        assert session.get(AnalysisRun, analysis_run_id) is None
        assert session.get(Fact, fact_id) is None
        assert session.get(TimelineEvent, timeline_event_id) is None
        assert session.get(Hypothesis, hypothesis_id) is None
        assert session.get(BiasFlag, bias_flag_id) is None
        assert session.get(RecommendedAction, action_id) is None

    with sqlite_engine.connect() as connection:
        association_count = connection.execute(
            select(func.count()).select_from(recommended_action_hypotheses),
        ).scalar_one()
        assert association_count == 0
