"""Typed, redacted-safe input contract for reviewed incident reports."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

from app.models.enums import (
    AnalysisRunStatus,
    ClaimSupportStatus,
    EvidenceType,
    FactReviewStatus,
    HypothesisStatus,
)
from app.schemas.ai_outputs import OpenQuestionSourceKind
from app.schemas.validation import EvidenceReferenceValidationStatus


class ReportInputModel(BaseModel):
    """Immutable base for data supplied to later report-generation stages."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ReportFactCategory(StrEnum):
    """Effective report placement after deterministic and human review."""

    CONFIRMED = "CONFIRMED"
    UNCONFIRMED = "UNCONFIRMED"


class ReportAssumptionSource(StrEnum):
    """Origin of an assumption included in the reviewed report input."""

    AI_IDENTIFIED = "AI_IDENTIFIED"
    HUMAN_RECLASSIFIED_FACT = "HUMAN_RECLASSIFIED_FACT"


class ReportIncidentInput(ReportInputModel):
    """Safe incident metadata needed to identify the report subject."""

    public_id: str
    name: str
    affected_service: str
    reported_start_time: datetime | None


class ReportAnalysisRunInput(ReportInputModel):
    """Safe audit metadata that deliberately excludes the raw response."""

    id: int
    status: AnalysisRunStatus
    provider_name: str
    model_name: str
    prompt_versions: dict[str, str]
    started_at: datetime
    completed_at: datetime | None


class ReportSummaryInput(ReportInputModel):
    """Reviewed report context from the validated summary stage."""

    text: str
    impact: str
    uncertainty: str
    unknowns: tuple[str, ...]


class ReportEvidenceReference(ReportInputModel):
    """Run-scoped evidence metadata without original or redacted evidence bodies."""

    evidence_code: str
    available: bool
    source_name: str | None
    evidence_type: EvidenceType | None


class ReportFactAIValues(ReportInputModel):
    """Original persisted AI fact values retained beside human review state."""

    claim: str
    support_status: ClaimSupportStatus
    confidence: int
    evidence_codes: tuple[str, ...]


class ReportFactInput(ReportInputModel):
    """One fact using its effective reviewed placement."""

    fact_id: int
    claim: str
    confidence: int
    category: ReportFactCategory
    human_status: FactReviewStatus
    evidence: tuple[ReportEvidenceReference, ...]
    ai_original: ReportFactAIValues


class ReportAssumptionAIValues(ReportInputModel):
    """Original AI values underlying an assumption."""

    claim: str
    reason: str | None
    required_evidence: tuple[str, ...]
    fact_support_status: ClaimSupportStatus | None
    fact_confidence: int | None
    fact_evidence_codes: tuple[str, ...]


class ReportAssumptionInput(ReportInputModel):
    """An AI-identified or human-reclassified assumption."""

    claim: str
    reason: str
    required_evidence: tuple[str, ...]
    source: ReportAssumptionSource
    originating_fact_id: int | None
    evidence: tuple[ReportEvidenceReference, ...]
    ai_original: ReportAssumptionAIValues


class ReportTimelineAIValues(ReportInputModel):
    """Original model timeline values before deterministic or human adjustment."""

    timestamp: str
    description: str
    confidence: int


class ReportTimelineEventInput(ReportInputModel):
    """One direct or inferred timeline event with explicit uncertainty."""

    event_id: int
    event_time: datetime | None
    description: str
    confidence: int
    is_inferred: bool
    uncertainty: str | None
    evidence: tuple[ReportEvidenceReference, ...]
    has_human_override: bool
    ai_original: ReportTimelineAIValues


class ReportHypothesisEvidenceInput(ReportInputModel):
    """Deterministically valid evidence used by the effective hypothesis."""

    reference: ReportEvidenceReference
    line_range: str
    relevance: str


class ReportHypothesisAIReferenceInput(ReportInputModel):
    """Original AI reference metadata plus its deterministic validation result."""

    evidence_code: str
    line_range: str
    relevance: str
    validation_status: EvidenceReferenceValidationStatus
    validation_message: str


class ReportHypothesisAIValues(ReportInputModel):
    """Original model hypothesis values retained for auditability."""

    title: str
    explanation: str
    confidence: int
    risk_of_acting: str
    supporting_evidence: tuple[ReportHypothesisAIReferenceInput, ...]
    contradicting_evidence: tuple[ReportHypothesisAIReferenceInput, ...]


class ReportHypothesisInput(ReportInputModel):
    """A reviewed hypothesis that cannot imply confirmation without human status."""

    hypothesis_id: int
    rank: int
    title: str
    explanation: str
    confidence: int
    validated_ai_confidence: int
    has_human_confidence_override: bool
    human_status: HypothesisStatus
    supporting_evidence: tuple[ReportHypothesisEvidenceInput, ...]
    contradicting_evidence: tuple[ReportHypothesisEvidenceInput, ...]
    missing_evidence: tuple[str, ...]
    validation_test: str
    expected_if_true: str
    expected_if_false: str
    ai_original: ReportHypothesisAIValues


class ReportActionInput(ReportInputModel):
    """A non-executing recommended investigation or mitigation action."""

    action_id: int
    description: str
    priority: str
    owner_role: str
    expected_information: str
    operational_risk: str
    evidence: tuple[ReportEvidenceReference, ...]
    linked_hypothesis_ranks: tuple[int, ...]


class ReportReasoningRiskInput(ReportInputModel):
    """A possible reasoning weakness retained with its mitigation."""

    risk_id: int
    name: str
    explanation: str
    trigger: str
    mitigation: str
    confidence: int


class ReportOpenQuestionInput(ReportInputModel):
    """An unresolved question and the evidence required to answer it."""

    question: str
    source_kind: OpenQuestionSourceKind
    source_reference: str
    rationale: str
    evidence_needed: tuple[str, ...]
    resolution_criteria: str


class ReportHumanNoteInput(ReportInputModel):
    """A human-authored note attached to the reviewed analysis run."""

    note_id: int
    note: str
    created_at: datetime


class ReportCriticFindingInput(ReportInputModel):
    """A safe adversarial finding included as an AI limitation."""

    concern: str
    affected_claim: str
    evidence: tuple[ReportEvidenceReference, ...]
    impact: str
    recommendation: str


class ReportValidationInput(ReportInputModel):
    """Deterministic limitations and unsupported-claim information."""

    claim_support_counts: dict[str, int]
    inferred_timeline_events: int
    hypotheses_with_contradictions: int
    unavailable_evidence_codes: tuple[str, ...]
    unsupported_fact_ids: tuple[int, ...]
    critic_findings: tuple[ReportCriticFindingInput, ...]
    critic_ignored_evidence: tuple[ReportEvidenceReference, ...]
    hypothesis_ranking_rationale: str


class ReportInput(ReportInputModel):
    """Complete reviewed, traceable, secret-safe input for report generation."""

    incident: ReportIncidentInput
    analysis_run: ReportAnalysisRunInput
    summary: ReportSummaryInput
    evidence: tuple[ReportEvidenceReference, ...]
    confirmed_facts: tuple[ReportFactInput, ...]
    unconfirmed_facts: tuple[ReportFactInput, ...]
    assumptions: tuple[ReportAssumptionInput, ...]
    timeline: tuple[ReportTimelineEventInput, ...]
    hypotheses: tuple[ReportHypothesisInput, ...]
    actions: tuple[ReportActionInput, ...]
    reasoning_risks: tuple[ReportReasoningRiskInput, ...]
    open_questions: tuple[ReportOpenQuestionInput, ...]
    human_notes: tuple[ReportHumanNoteInput, ...]
    validation: ReportValidationInput


class ReportDraftUpdate(BaseModel):
    """Validated human edit submitted from the report preview."""

    editable_text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=200_000),
    ]

    model_config = ConfigDict(extra="forbid", frozen=True)
