"""Provider-neutral AI interface and settings-driven provider selection."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Never, Protocol, runtime_checkable

from pydantic import ValidationError

from app.config import Settings
from app.schemas.ai_outputs import (
    AIOutput,
    CriticOutputV1,
    HypothesesOutputV1,
    ReasoningRisksOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (
    AIFailureCategory,
    AIFailureDetails,
    AIRequest,
    AIResult,
    AIResultMetadata,
    FailureAuditData,
    OutputSchemaIdentifier,
    SuccessAuditData,
)


class AIProviderName(StrEnum):
    """Concrete provider selections supported by the application factory."""

    FAKE = "fake"
    GEMINI = "gemini"


@runtime_checkable
class AIProvider(Protocol):
    """Provider-neutral structured generation boundary."""

    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        """Generate one locally validated result for a typed request."""
        ...


class AIProviderConfigurationError(ValueError):
    """Safe configuration failure raised before provider construction."""

    def __init__(self, explanation: str) -> None:
        self.details = AIFailureDetails(
            category=AIFailureCategory.CONFIGURATION,
            explanation=explanation,
        )
        super().__init__(self.details.explanation)

    def __repr__(self) -> str:
        """Return only the safe provider-neutral failure details."""
        return (
            f"{type(self).__name__}("
            f"category={self.details.category.value!r}, "
            f"explanation={self.details.explanation!r})"
        )


class AIProviderExecutionError(RuntimeError):
    """Safe provider-neutral failure with internal-only audit details."""

    def __init__(self, details: AIFailureDetails) -> None:
        self.details = details
        super().__init__(details.explanation)

    def __repr__(self) -> str:
        """Exclude provider responses and other internal audit fields."""
        return (
            f"{type(self).__name__}("
            f"category={self.details.category.value!r}, "
            f"request_identifier={self.details.request_identifier!r}, "
            f"explanation={self.details.explanation!r})"
        )


@dataclass(frozen=True, slots=True)
class StructuredResponseOutcome:
    """Internal result of provider-neutral response extraction and validation."""

    output: AIOutput | None
    failure_category: AIFailureCategory | None

    def __post_init__(self) -> None:
        has_output = self.output is not None
        has_failure = self.failure_category is not None
        if has_output == has_failure:
            raise ValueError(
                "structured response outcome requires one output or failure"
            )


@dataclass(frozen=True, slots=True)
class BoundedRetryPolicy:
    """Provider-neutral total-attempt limit and deterministic backoff schedule."""

    max_attempts: int
    retry_delay_seconds: int | float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 1
        ):
            raise AIProviderConfigurationError(
                "AI provider maximum attempts must be a positive integer."
            )
        retry_delay = self.retry_delay_seconds
        if isinstance(retry_delay, bool) or not isinstance(retry_delay, (int, float)):
            raise AIProviderConfigurationError(
                "AI provider retry delay must be a non-negative number."
            )
        if retry_delay < 0 or (
            isinstance(retry_delay, float) and not isfinite(retry_delay)
        ):
            raise AIProviderConfigurationError(
                "AI provider retry delay must be a non-negative number."
            )

    @property
    def attempt_numbers(self) -> range:
        """Return the finite sequence of allowed total attempt numbers."""
        return range(1, self.max_attempts + 1)

    def has_next_attempt(self, attempt_count: int) -> bool:
        """Return whether another attempt remains after the current attempt."""
        return attempt_count < self.max_attempts

    def delay_before_next_attempt(self, attempt_count: int) -> float:
        """Return deterministic linear backoff for the next allowed attempt."""
        if not self.has_next_attempt(attempt_count):
            raise ValueError("no AI provider retry remains")
        return float(self.retry_delay_seconds) * attempt_count


_OUTPUT_MODELS: dict[OutputSchemaIdentifier, type[AIOutput]] = {
    OutputSchemaIdentifier.SUMMARY_V1: SummaryOutputV1,
    OutputSchemaIdentifier.TIMELINE_V1: TimelineOutputV1,
    OutputSchemaIdentifier.HYPOTHESES_V1: HypothesesOutputV1,
    OutputSchemaIdentifier.CRITIC_V1: CriticOutputV1,
    OutputSchemaIdentifier.REASONING_RISKS_V1: ReasoningRisksOutputV1,
}

_SAFE_FAILURE_EXPLANATIONS = {
    AIFailureCategory.TIMEOUT: "The AI provider request timed out.",
    AIFailureCategory.RATE_LIMIT: "The AI provider is temporarily rate limited.",
    AIFailureCategory.TRANSIENT_PROVIDER_FAILURE: (
        "The AI provider request failed safely."
    ),
    AIFailureCategory.AUTHENTICATION: ("The AI provider rejected its credentials."),
    AIFailureCategory.MALFORMED_JSON: ("The AI provider returned malformed JSON."),
    AIFailureCategory.SCHEMA_VALIDATION: (
        "The AI provider response failed schema validation."
    ),
    AIFailureCategory.UNSUPPORTED_OUTPUT_SCHEMA: (
        "The AI provider does not support the requested output schema."
    ),
    AIFailureCategory.UNKNOWN_PROMPT: (
        "The requested AI prompt could not be resolved."
    ),
    AIFailureCategory.EXHAUSTED_RETRIES: (
        "The AI provider failed after the allowed attempts."
    ),
}


def select_output_model(request: AIRequest) -> type[AIOutput]:
    """Select an allowlisted local output model or fail before a provider call."""
    try:
        output_model = _OUTPUT_MODELS.get(request.output_schema)
    except TypeError:
        output_model = None
    if output_model is None:
        raise_ai_provider_failure(
            request=request,
            category=AIFailureCategory.UNSUPPORTED_OUTPUT_SCHEMA,
            attempt_count=1,
        )
    return output_model


def process_structured_response(
    raw_response: str | None,
    output_model: type[AIOutput],
) -> StructuredResponseOutcome:
    """Extract, parse, and locally validate one provider response."""
    if raw_response is None or not isinstance(raw_response, str):
        return StructuredResponseOutcome(
            output=None,
            failure_category=AIFailureCategory.TRANSIENT_PROVIDER_FAILURE,
        )
    try:
        response_data = json.loads(raw_response)
    except json.JSONDecodeError:
        return StructuredResponseOutcome(
            output=None,
            failure_category=AIFailureCategory.MALFORMED_JSON,
        )
    try:
        output = output_model.model_validate(response_data)
    except ValidationError:
        return StructuredResponseOutcome(
            output=None,
            failure_category=AIFailureCategory.SCHEMA_VALIDATION,
        )
    return StructuredResponseOutcome(
        output=output,
        failure_category=None,
    )


def build_ai_result(
    *,
    request: AIRequest,
    output: AIOutput,
    provider_name: str,
    model_name: str,
    attempt_count: int,
    raw_response: str,
) -> AIResult[AIOutput]:
    """Build one provider-neutral result with internal-only success audit data."""
    return AIResult[AIOutput](
        output=output,
        metadata=AIResultMetadata(
            provider_name=provider_name,
            model_name=model_name,
            system_prompt=request.prompts.system,
            task_prompt=request.prompts.task,
            analysis_stage=request.metadata.analysis_stage,
            output_schema=request.output_schema,
            request_identifier=request.metadata.request_identifier,
            attempt_count=attempt_count,
        ),
        audit=SuccessAuditData(raw_response=raw_response),
    )


def raise_ai_provider_failure(
    *,
    request: AIRequest,
    category: AIFailureCategory,
    attempt_count: int,
    raw_response: str | None = None,
) -> Never:
    """Raise a sanitized provider-neutral failure with internal-only audit data."""
    explanation = _SAFE_FAILURE_EXPLANATIONS.get(
        category,
        "The AI provider request failed safely.",
    )
    audit = FailureAuditData(
        request_identifier=request.metadata.request_identifier,
        attempt_count=attempt_count,
        raw_response=raw_response,
    )
    raise AIProviderExecutionError(
        AIFailureDetails(
            category=category,
            request_identifier=request.metadata.request_identifier,
            explanation=explanation,
            audit=audit,
        )
    ) from None


ProviderBuilder = Callable[[Settings], AIProvider]


class AIProviderFactory:
    """Select and construct an injected provider implementation from settings."""

    def __init__(
        self,
        *,
        fake_builder: ProviderBuilder | None = None,
        gemini_builder: ProviderBuilder | None = None,
    ) -> None:
        self._builders = {
            AIProviderName.FAKE: fake_builder,
            AIProviderName.GEMINI: gemini_builder,
        }

    def create(self, settings: Settings) -> AIProvider:
        """Return the selected provider after safe fail-fast validation."""
        provider_name = self._parse_provider_name(settings.ai_provider)
        if provider_name is AIProviderName.GEMINI:
            self._validate_gemini_configuration(settings)

        builder = self._builders[provider_name]
        if builder is None:
            raise AIProviderConfigurationError(
                "The selected AI provider implementation is not registered."
            )

        provider = builder(settings)
        if not isinstance(provider, AIProvider):
            raise AIProviderConfigurationError(
                "The selected AI provider implementation is invalid."
            )
        return provider

    @staticmethod
    def _parse_provider_name(value: str) -> AIProviderName:
        try:
            return AIProviderName(value)
        except ValueError:
            raise AIProviderConfigurationError(
                "AI_PROVIDER must be either 'fake' or 'gemini'."
            ) from None

    @staticmethod
    def _validate_gemini_configuration(settings: Settings) -> None:
        missing_settings: list[str] = []
        api_key = settings.gemini_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            missing_settings.append("GEMINI_API_KEY")
        if settings.gemini_model is None or not settings.gemini_model.strip():
            missing_settings.append("GEMINI_MODEL")

        if missing_settings:
            setting_names = " and ".join(missing_settings)
            raise AIProviderConfigurationError(
                f"Gemini provider configuration requires {setting_names}."
            )
