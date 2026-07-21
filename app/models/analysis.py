"""Auditable analysis-run and structured reasoning persistence models."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    Enum as SQLAlchemyEnum,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.enums import (
    AnalysisRunStatus,
    ClaimSupportStatus,
    FactReviewStatus,
    HypothesisStatus,
)
from app.models.mixins import utc_now
from app.models.types import UTCDateTime

if TYPE_CHECKING:
    from app.models.incident import Incident


MODEL_NAME_MAX_LENGTH = 200
PROVIDER_NAME_MAX_LENGTH = 100
HYPOTHESIS_TITLE_MAX_LENGTH = 255
BIAS_TYPE_MAX_LENGTH = 100
ACTION_PRIORITY_MAX_LENGTH = 50
OWNER_ROLE_MAX_LENGTH = 100


recommended_action_hypotheses = Table(
    "recommended_action_hypotheses",
    Base.metadata,
    Column(
        "recommended_action_id",
        ForeignKey("recommended_actions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "hypothesis_id",
        ForeignKey("hypotheses.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class AnalysisRun(Base):
    """One auditable execution of the structured analysis pipeline."""

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(
        String(MODEL_NAME_MAX_LENGTH),
        nullable=False,
    )
    provider_name: Mapped[str] = mapped_column(
        String(PROVIDER_NAME_MAX_LENGTH),
        nullable=False,
    )
    prompt_versions: Mapped[dict[str, str]] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )
    input_evidence_codes: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )
    raw_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[AnalysisRunStatus] = mapped_column(
        SQLAlchemyEnum(
            AnalysisRunStatus,
            name="analysis_run_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
        ),
        default=AnalysisRunStatus.RUNNING,
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=utc_now,
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )

    incident: Mapped[Incident] = relationship(back_populates="analysis_runs")
    facts: Mapped[list[Fact]] = relationship(
        back_populates="analysis_run",
        cascade="all, delete-orphan",
        single_parent=True,
    )
    timeline_events: Mapped[list[TimelineEvent]] = relationship(
        back_populates="analysis_run",
        cascade="all, delete-orphan",
        single_parent=True,
    )
    hypotheses: Mapped[list[Hypothesis]] = relationship(
        back_populates="analysis_run",
        cascade="all, delete-orphan",
        single_parent=True,
    )
    bias_flags: Mapped[list[BiasFlag]] = relationship(
        back_populates="analysis_run",
        cascade="all, delete-orphan",
        single_parent=True,
    )
    actions: Mapped[list[RecommendedAction]] = relationship(
        back_populates="analysis_run",
        cascade="all, delete-orphan",
        single_parent=True,
    )


class Fact(Base):
    """A model-generated claim with evidence support and human review state."""

    __tablename__ = "facts"
    __table_args__ = (
        CheckConstraint(
            "confidence BETWEEN 0 AND 100",
            name="ck_facts_confidence_range",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    support_status: Mapped[ClaimSupportStatus] = mapped_column(
        SQLAlchemyEnum(
            ClaimSupportStatus,
            name="claim_support_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
        ),
        nullable=False,
    )
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)
    evidence_codes: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )
    supporting_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    human_status: Mapped[FactReviewStatus] = mapped_column(
        SQLAlchemyEnum(
            FactReviewStatus,
            name="fact_review_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
        ),
        default=FactReviewStatus.PENDING,
        nullable=False,
    )

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="facts")


class TimelineEvent(Base):
    """A direct or inferred event in an incident timeline."""

    __tablename__ = "timeline_events"
    __table_args__ = (
        CheckConstraint(
            "confidence BETWEEN 0 AND 100",
            name="ck_timeline_events_confidence_range",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    event_time: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_codes: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )
    is_inferred: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)

    analysis_run: Mapped[AnalysisRun] = relationship(
        back_populates="timeline_events",
    )


class Hypothesis(Base):
    """A ranked possible explanation that requires human validation."""

    __tablename__ = "hypotheses"
    __table_args__ = (
        CheckConstraint("rank > 0", name="ck_hypotheses_positive_rank"),
        CheckConstraint(
            "confidence BETWEEN 0 AND 100",
            name="ck_hypotheses_confidence_range",
        ),
        UniqueConstraint(
            "analysis_run_id",
            "rank",
            name="uq_hypotheses_run_rank",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(
        String(HYPOTHESIS_TITLE_MAX_LENGTH),
        nullable=False,
    )
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)
    supporting_evidence_codes: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )
    contradicting_evidence_codes: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )
    missing_evidence: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )
    recommended_test: Mapped[str] = mapped_column(Text, nullable=False)
    expected_true_result: Mapped[str] = mapped_column(Text, nullable=False)
    expected_false_result: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[HypothesisStatus] = mapped_column(
        SQLAlchemyEnum(
            HypothesisStatus,
            name="hypothesis_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
        ),
        default=HypothesisStatus.UNTESTED,
        nullable=False,
    )

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="hypotheses")
    recommended_actions: Mapped[list[RecommendedAction]] = relationship(
        secondary=recommended_action_hypotheses,
        back_populates="hypotheses",
    )


class BiasFlag(Base):
    """A possible reasoning risk detected in an analysis run."""

    __tablename__ = "bias_flags"
    __table_args__ = (
        CheckConstraint(
            "confidence BETWEEN 0 AND 100",
            name="ck_bias_flags_confidence_range",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    bias_type: Mapped[str] = mapped_column(
        String(BIAS_TYPE_MAX_LENGTH),
        nullable=False,
    )
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    trigger: Mapped[str] = mapped_column(Text, nullable=False)
    mitigation: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="bias_flags")


class RecommendedAction(Base):
    """A non-executing investigation or mitigation recommendation."""

    __tablename__ = "recommended_actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(
        String(ACTION_PRIORITY_MAX_LENGTH),
        nullable=False,
    )
    evidence_codes: Mapped[list[str]] = mapped_column(
        MutableList.as_mutable(JSON),
        default=list,
        nullable=False,
    )
    owner_role: Mapped[str] = mapped_column(
        String(OWNER_ROLE_MAX_LENGTH),
        nullable=False,
    )
    expected_information: Mapped[str] = mapped_column(Text, nullable=False)
    operational_risk: Mapped[str] = mapped_column(Text, nullable=False)

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="actions")
    hypotheses: Mapped[list[Hypothesis]] = relationship(
        secondary=recommended_action_hypotheses,
        back_populates="recommended_actions",
    )
