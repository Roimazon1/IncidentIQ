"""Strict provider-neutral contracts for the IncidentIQ AI boundary."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Generic, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    model_validator,
)

from app.models.enums import ClaimSupportStatus
from app.schemas.ai_outputs import (
    ConfidenceScore,
    CriticOutputV1,
    EvidenceReferenceV1,
    HypothesisIdentifier,
    HypothesisValidationTestV1,
    HypothesesOutputV1,
    NonBlankText,
    ReasoningRisksOutputV1,
    PositiveRank,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.evidence import EvidenceManifest, Sha256Checksum
from app.schemas.incident import IncidentPublicId


SafeIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]


def _require_log_safe_name(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError("name must contain only log-safe characters")
    return value


LogSafeName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=200),
    AfterValidator(_require_log_safe_name),
]
SafeExplanation = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]


class StrictAIContract(BaseModel):
    """Immutable base for provider-neutral request, result, and audit data."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PromptName(StrEnum):
    """Allowlisted prompt names owned by the centralized registry."""

    SYSTEM = "system"
    SUMMARY = "summary"
    TIMELINE = "timeline"
    HYPOTHESES = "hypotheses"
    CRITIC = "critic"
    BIAS = "bias"
    OPEN_QUESTIONS = "open_questions"
    POSTMORTEM = "postmortem"


class PromptVersion(StrEnum):
    """Explicit prompt versions accepted at the typed provider boundary."""

    V1 = "v1"


class AnalysisStage(StrEnum):
    """Provider-neutral stages that may request registered prompts."""

    SYSTEM = "system"
    SUMMARY = "summary"
    TIMELINE = "timeline"
    HYPOTHESES = "hypotheses"
    CRITIC = "critic"
    BIAS = "bias"
    OPEN_QUESTIONS = "open_questions"
    POSTMORTEM = "postmortem"


def _require_requestable_analysis_stage(value: AnalysisStage) -> AnalysisStage:
    if value is AnalysisStage.SYSTEM:
        raise ValueError("system is not a requestable analysis stage")
    return value


RequestableAnalysisStage = Annotated[
    AnalysisStage,
    AfterValidator(_require_requestable_analysis_stage),
]


class OutputSchemaIdentifier(StrEnum):
    """Allowlisted local output schemas selectable by a provider request."""

    SUMMARY_V1 = "summary_v1"
    TIMELINE_V1 = "timeline_v1"
    HYPOTHESES_V1 = "hypotheses_v1"
    CRITIC_V1 = "critic_v1"
    REASONING_RISKS_V1 = "reasoning_risks_v1"
    OPEN_QUESTIONS_V1 = "open_questions_v1"


class PromptReference(StrictAIContract):
    """Typed reference to one prompt registered by name and version."""

    name: PromptName
    version: PromptVersion


class PromptBundle(StrictAIContract):
    """Exactly one system prompt and one task prompt reference."""

    system: PromptReference
    task: PromptReference

    @model_validator(mode="after")
    def validate_prompt_roles(self) -> PromptBundle:
        """Reject reversed or system-only task prompt references."""
        if self.system.name is not PromptName.SYSTEM:
            raise ValueError("system prompt must reference PromptName.SYSTEM")
        if self.task.name is PromptName.SYSTEM:
            raise ValueError("task prompt must not reference PromptName.SYSTEM")
        return self


class SafeAIMetadata(StrictAIContract):
    """Allowlisted non-sensitive identifiers used for request traceability."""

    request_identifier: SafeIdentifier
    incident_public_identifier: IncidentPublicId
    analysis_stage: RequestableAnalysisStage
    evidence_manifest_checksum: Sha256Checksum | None = None


class CriticContextV1(StrictAIContract):
    """Validated initial analysis supplied separately to the critic stage."""

    summary: SummaryOutputV1
    timeline: TimelineOutputV1
    hypotheses: HypothesesOutputV1


class EvidenceReferenceValidationStatus(StrEnum):
    """Provider-neutral deterministic outcomes for generated citations."""

    VALID = "valid"
    UNKNOWN_EVIDENCE_ID = "unknown_evidence_id"
    INVALID_LINE_RANGE = "invalid_line_range"
    EXCERPT_MISMATCH = "excerpt_mismatch"


class ValidatedEvidenceReferenceV1(StrictAIContract):
    """One generated reference and its deterministic validation outcome."""

    reference: EvidenceReferenceV1
    status: EvidenceReferenceValidationStatus
    message: NonBlankText


class ValidatedFactV1(StrictAIContract):
    """A fact candidate with deterministic support classification."""

    claim: NonBlankText
    confidence: ConfidenceScore
    support_status: ClaimSupportStatus
    evidence: tuple[ValidatedEvidenceReferenceV1, ...] = Field(min_length=1)


class ValidatedTimelineEventV1(StrictAIContract):
    """A timeline event carrying its deterministic persisted confidence."""

    timestamp: NonBlankText
    description: NonBlankText
    evidence: tuple[ValidatedEvidenceReferenceV1, ...] = Field(min_length=1)
    is_inferred: StrictBool
    persisted_confidence: ConfidenceScore
    uncertainty_explanation: NonBlankText | None = None


class ValidatedHypothesisEvidenceV1(StrictAIContract):
    """Hypothesis evidence with its deterministic reference outcome."""

    reference: ValidatedEvidenceReferenceV1
    relevance: NonBlankText


class ValidatedHypothesisV1(StrictAIContract):
    """A hypothesis with deterministic confidence and citation outcomes."""

    hypothesis_id: HypothesisIdentifier
    rank: PositiveRank
    title: NonBlankText
    explanation: NonBlankText
    adjusted_confidence: ConfidenceScore
    supporting_evidence: tuple[ValidatedHypothesisEvidenceV1, ...] = Field(min_length=1)
    contradicting_evidence: tuple[ValidatedHypothesisEvidenceV1, ...]
    missing_evidence: tuple[NonBlankText, ...]
    validation_test: HypothesisValidationTestV1
    risk_of_acting: NonBlankText


class ValidatedAnalysisViewV1(StrictAIContract):
    """Deterministic P7 validation state safe for provider reasoning."""

    facts: tuple[ValidatedFactV1, ...]
    timeline: tuple[ValidatedTimelineEventV1, ...]
    hypotheses: tuple[ValidatedHypothesisV1, ...]


class BiasContextV1(StrictAIContract):
    """Clearly separated original, validated, and critic analysis views."""

    original_analysis: CriticContextV1
    validated_analysis: ValidatedAnalysisViewV1
    critic: CriticOutputV1


class OpenQuestionsContextV1(StrictAIContract):
    """Validated analysis, critic, and bias results for open questions."""

    analysis_context: BiasContextV1
    reasoning_risks: ReasoningRisksOutputV1


class AIRequest(StrictAIContract):
    """Typed, provider-neutral input accepted by an IncidentIQ AI provider."""

    evidence_manifest: EvidenceManifest
    prompts: PromptBundle
    output_schema: OutputSchemaIdentifier
    metadata: SafeAIMetadata
    critic_context: CriticContextV1 | None = None
    bias_context: BiasContextV1 | None = None
    open_questions_context: OpenQuestionsContextV1 | None = None

    @model_validator(mode="after")
    def validate_analysis_context_roles(self) -> AIRequest:
        """Require each typed analysis context only for its owning stage."""
        is_critic_request = self.metadata.analysis_stage is AnalysisStage.CRITIC
        is_bias_request = self.metadata.analysis_stage is AnalysisStage.BIAS
        is_open_questions_request = (
            self.metadata.analysis_stage is AnalysisStage.OPEN_QUESTIONS
        )
        if is_critic_request and self.critic_context is None:
            raise ValueError(
                "critic requests require validated initial analysis context"
            )
        if not is_critic_request and self.critic_context is not None:
            raise ValueError("critic context is only accepted for critic requests")
        if is_bias_request and self.bias_context is None:
            raise ValueError("bias requests require validated analysis context")
        if not is_bias_request and self.bias_context is not None:
            raise ValueError("bias context is only accepted for bias requests")
        if is_open_questions_request and self.open_questions_context is None:
            raise ValueError(
                "open-question requests require validated reasoning context"
            )
        if not is_open_questions_request and self.open_questions_context is not None:
            raise ValueError(
                "open-question context is only accepted for open-question requests"
            )
        return self


class AIResultMetadata(StrictAIContract):
    """Safe provider-neutral traceability metadata for a completed attempt."""

    provider_name: LogSafeName
    model_name: LogSafeName
    system_prompt: PromptReference
    task_prompt: PromptReference
    analysis_stage: RequestableAnalysisStage
    output_schema: OutputSchemaIdentifier
    request_identifier: SafeIdentifier
    attempt_count: int = Field(strict=True, gt=0)


class SuccessAuditData(StrictAIContract):
    """Internal exact provider response retained after successful validation."""

    raw_response: str = Field(exclude=True, repr=False)


class FailureAuditData(StrictAIContract):
    """Internal latest response retained when provider processing fails."""

    request_identifier: SafeIdentifier
    attempt_count: int = Field(strict=True, gt=0)
    raw_response: str | None = Field(default=None, exclude=True, repr=False)


class AIFailureCategory(StrEnum):
    """Safe provider-neutral failure categories exposed across boundaries."""

    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    TRANSIENT_PROVIDER_FAILURE = "transient_provider_failure"
    AUTHENTICATION = "authentication"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_VALIDATION = "schema_validation"
    UNSUPPORTED_OUTPUT_SCHEMA = "unsupported_output_schema"
    UNKNOWN_PROMPT = "unknown_prompt"
    EXHAUSTED_RETRIES = "exhausted_retries"


class AIFailureDetails(StrictAIContract):
    """Safe public failure fields plus non-serializable internal audit data."""

    category: AIFailureCategory
    request_identifier: SafeIdentifier | None = None
    explanation: SafeExplanation
    audit: FailureAuditData | None = Field(default=None, exclude=True, repr=False)


AIOutputT = TypeVar("AIOutputT", bound=BaseModel)


class AIResult(StrictAIContract, Generic[AIOutputT]):
    """Validated output with safe metadata and internal-only success audit."""

    output: AIOutputT
    metadata: AIResultMetadata
    audit: SuccessAuditData = Field(exclude=True, repr=False)
