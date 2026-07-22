"""Provider-neutral, versioned Pydantic contracts for AI output."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    model_validator,
)

from app.schemas.evidence import EvidenceCode, LineRange


NonBlankText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
ConfidenceScore = Annotated[int, Field(strict=True, ge=0, le=100)]
PositiveRank = Annotated[int, Field(strict=True, gt=0)]
HypothesisIdentifier = Annotated[
    str,
    StringConstraints(pattern=r"^H-\d{3}$"),
]


class StrictAIOutputModel(BaseModel):
    """Immutable base contract for model-generated structured output."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionPriority(StrEnum):
    """Allowlisted urgency values for non-executing recommended actions."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EvidenceReferenceV1(StrictAIOutputModel):
    """A typed citation to one traceable location in redacted evidence."""

    evidence_id: EvidenceCode
    line_range: LineRange
    excerpt: NonBlankText | None = None


class SummaryAndImpactV1(StrictAIOutputModel):
    """Neutral incident summary with explicit impact and uncertainty."""

    text: NonBlankText
    impact: NonBlankText
    uncertainty: NonBlankText


class FactItemV1(StrictAIOutputModel):
    """A factual claim that cites at least one typed evidence location."""

    claim: NonBlankText
    evidence: tuple[EvidenceReferenceV1, ...] = Field(min_length=1)
    confidence: ConfidenceScore


class AssumptionItemV1(StrictAIOutputModel):
    """An explicitly unproven statement and the evidence needed to test it."""

    claim: NonBlankText
    reason: NonBlankText
    required_evidence: tuple[NonBlankText, ...] = Field(min_length=1)


class SummaryOutputV1(StrictAIOutputModel):
    """Version-one summary-stage output."""

    summary: SummaryAndImpactV1
    facts: tuple[FactItemV1, ...]
    assumptions: tuple[AssumptionItemV1, ...]
    unknowns: tuple[NonBlankText, ...]


class TimelineEventV1(StrictAIOutputModel):
    """A direct or inferred event with traceable evidence and uncertainty."""

    timestamp: NonBlankText
    description: NonBlankText
    evidence: tuple[EvidenceReferenceV1, ...] = Field(min_length=1)
    is_inferred: StrictBool
    confidence: ConfidenceScore
    uncertainty_explanation: NonBlankText | None = None

    @model_validator(mode="after")
    def require_inference_explanation(self) -> TimelineEventV1:
        """Require explicit uncertainty context for every inferred event."""
        if self.is_inferred and self.uncertainty_explanation is None:
            raise ValueError(
                "inferred timeline events require an uncertainty explanation"
            )
        return self


class TimelineOutputV1(StrictAIOutputModel):
    """Version-one timeline-stage output."""

    events: tuple[TimelineEventV1, ...]


class SupportingEvidenceV1(StrictAIOutputModel):
    """Evidence that raises the plausibility of a hypothesis."""

    reference: EvidenceReferenceV1
    relevance: NonBlankText


class ContradictingEvidenceV1(StrictAIOutputModel):
    """Evidence that weakens or conflicts with a hypothesis."""

    reference: EvidenceReferenceV1
    relevance: NonBlankText


class HypothesisValidationTestV1(StrictAIOutputModel):
    """A non-executing test that can strengthen or weaken a hypothesis."""

    description: NonBlankText
    expected_if_true: NonBlankText
    expected_if_false: NonBlankText


class HypothesisV1(StrictAIOutputModel):
    """One ranked, explicitly unconfirmed root-cause hypothesis."""

    hypothesis_id: HypothesisIdentifier
    rank: PositiveRank
    title: NonBlankText
    explanation: NonBlankText
    confidence: ConfidenceScore
    supporting_evidence: tuple[SupportingEvidenceV1, ...] = Field(min_length=1)
    contradicting_evidence: tuple[ContradictingEvidenceV1, ...]
    missing_evidence: tuple[NonBlankText, ...]
    validation_test: HypothesisValidationTestV1
    risk_of_acting: NonBlankText


def _validate_hypothesis_ranking(
    hypotheses: tuple[HypothesisV1, ...],
) -> tuple[HypothesisV1, ...]:
    hypothesis_ids = [hypothesis.hypothesis_id for hypothesis in hypotheses]
    if len(set(hypothesis_ids)) != len(hypothesis_ids):
        raise ValueError("hypothesis identifiers must be unique")

    ranks = [hypothesis.rank for hypothesis in hypotheses]
    if len(set(ranks)) != len(ranks):
        raise ValueError("hypothesis ranks must be unique")
    if sorted(ranks) != list(range(1, len(hypotheses) + 1)):
        raise ValueError("hypothesis ranks must form the contiguous sequence 1..N")

    return hypotheses


RankedHypotheses = Annotated[
    tuple[HypothesisV1, ...],
    Field(min_length=3),
    AfterValidator(_validate_hypothesis_ranking),
]


class HypothesesOutputV1(StrictAIOutputModel):
    """Version-one multi-hypothesis stage output."""

    hypotheses: RankedHypotheses


class RecommendedActionV1(StrictAIOutputModel):
    """A recommended human action that the system must not execute."""

    description: NonBlankText
    priority: ActionPriority
    linked_hypothesis_ids: tuple[HypothesisIdentifier, ...] = Field(min_length=1)
    evidence: tuple[EvidenceReferenceV1, ...] = Field(min_length=1)
    owner_role: NonBlankText
    expected_information: NonBlankText
    operational_risk: NonBlankText


class ActionsOutputV1(StrictAIOutputModel):
    """Version-one recommended-action output."""

    actions: tuple[RecommendedActionV1, ...]


class ReasoningRiskV1(StrictAIOutputModel):
    """A possible reasoning risk phrased as a warning, not an accusation."""

    name: NonBlankText
    location: NonBlankText
    trigger: NonBlankText
    potential_effect: NonBlankText
    mitigation: NonBlankText
    confidence: ConfidenceScore


class ReasoningRisksOutputV1(StrictAIOutputModel):
    """Version-one bias and fallacy analysis output."""

    risks: tuple[ReasoningRiskV1, ...]


class OpenQuestionSourceKind(StrEnum):
    """Allowlisted unresolved analysis elements an open question may trace to."""

    UNRESOLVED_CLAIM = "UNRESOLVED_CLAIM"
    HYPOTHESIS = "HYPOTHESIS"
    CONTRADICTION = "CONTRADICTION"
    ASSUMPTION = "ASSUMPTION"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"


class OpenQuestionV1(StrictAIOutputModel):
    """An actionable unresolved question and the evidence needed to answer it."""

    question: NonBlankText
    source_kind: OpenQuestionSourceKind
    source_reference: NonBlankText
    rationale: NonBlankText
    evidence_needed: tuple[NonBlankText, ...] = Field(min_length=1)
    resolution_criteria: NonBlankText


class OpenQuestionsOutputV1(StrictAIOutputModel):
    """Version-one actionable open-question output."""

    questions: tuple[OpenQuestionV1, ...] = Field(min_length=1)


class CriticFindingV1(StrictAIOutputModel):
    """One adversarial finding about a potentially weak conclusion."""

    concern: NonBlankText
    affected_claim: NonBlankText
    evidence: tuple[EvidenceReferenceV1, ...]
    impact: NonBlankText
    recommendation: NonBlankText


class CriticOutputV1(StrictAIOutputModel):
    """Version-one adversarial critique that preserves the original result."""

    findings: tuple[CriticFindingV1, ...] = Field(min_length=1)
    ignored_evidence: tuple[EvidenceReferenceV1, ...]
    alternative_hypothesis: HypothesisV1 | None
    ranking_rationale: NonBlankText


class CompleteAnalysisOutputV1(StrictAIOutputModel):
    """Composition of every validated version-one analysis output."""

    summary: SummaryAndImpactV1
    facts: tuple[FactItemV1, ...]
    assumptions: tuple[AssumptionItemV1, ...]
    timeline: tuple[TimelineEventV1, ...]
    hypotheses: RankedHypotheses
    actions: tuple[RecommendedActionV1, ...]
    open_questions: tuple[OpenQuestionV1, ...]
    reasoning_risks: tuple[ReasoningRiskV1, ...]
    critic: CriticOutputV1


AIOutput: TypeAlias = (
    SummaryOutputV1
    | TimelineOutputV1
    | HypothesesOutputV1
    | ActionsOutputV1
    | ReasoningRisksOutputV1
    | OpenQuestionsOutputV1
    | CriticOutputV1
    | CompleteAnalysisOutputV1
)
