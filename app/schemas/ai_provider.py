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
    StringConstraints,
    model_validator,
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


class AIRequest(StrictAIContract):
    """Redacted-only input accepted by any IncidentIQ AI provider."""

    evidence_manifest: EvidenceManifest
    prompts: PromptBundle
    output_schema: OutputSchemaIdentifier
    metadata: SafeAIMetadata


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
