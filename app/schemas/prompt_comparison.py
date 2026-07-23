"""Sanitized structured results for deterministic prompt comparison."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.ai_outputs import (
    CriticOutputV1,
    HypothesesOutputV1,
)
from app.schemas.ai_provider import (
    EvidenceReferenceValidationStatus,
    OutputSchemaIdentifier,
    PromptReference,
    ValidatedHypothesisV1,
)
from app.schemas.evidence import EvidenceCode, LineRange


class PromptComparisonVariantName(StrEnum):
    """Allowlisted evaluation variants required by P10-03."""

    NEUTRAL_EVIDENCE_FIRST = "neutral_evidence_first"
    LEADING_DEPLOYMENT_V2_4_1 = "leading_deployment_v2_4_1"
    ADVERSARIAL_TOP_HYPOTHESIS = "adversarial_top_hypothesis"


class StrictPromptComparisonModel(BaseModel):
    """Immutable base for comparison output that excludes provider audit data."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ComparisonEvidenceValidation(StrictPromptComparisonModel):
    """One deterministic evidence-reference validation outcome."""

    evidence_id: EvidenceCode
    line_range: LineRange
    status: EvidenceReferenceValidationStatus
    message: str = Field(min_length=1)


class HypothesisComparisonVariant(StrictPromptComparisonModel):
    """One typed hypothesis-generation result and its validated evidence view."""

    variant: PromptComparisonVariantName
    task_prompt: PromptReference
    output_schema: OutputSchemaIdentifier
    hypotheses: HypothesesOutputV1
    validated_hypotheses: tuple[ValidatedHypothesisV1, ...] = Field(min_length=3)


class AdversarialComparisonVariant(StrictPromptComparisonModel):
    """One typed challenge of the neutral result's top-ranked hypothesis."""

    variant: PromptComparisonVariantName
    task_prompt: PromptReference
    output_schema: OutputSchemaIdentifier
    challenged_hypothesis_id: str = Field(min_length=1)
    critique: CriticOutputV1
    evidence_validation: tuple[ComparisonEvidenceValidation, ...]


class PromptComparisonResult(StrictPromptComparisonModel):
    """Safe side-by-side data for later reflective evaluation."""

    incident_public_id: str = Field(min_length=1)
    provider_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    evidence_codes: tuple[EvidenceCode, ...] = Field(min_length=1)
    neutral: HypothesisComparisonVariant
    leading: HypothesisComparisonVariant
    adversarial: AdversarialComparisonVariant
