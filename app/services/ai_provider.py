"""Provider-neutral AI interface and settings-driven provider selection."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Never, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.schemas.ai_outputs import (
    AIOutput,
    CriticOutputV1,
    HypothesisV1,
    HypothesesOutputV1,
    OpenQuestionV1,
    OpenQuestionsOutputV1,
    PostmortemOutputV1,
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
    AnalysisStage,
    FailureAuditData,
    OutputSchemaIdentifier,
    PromptBundle,
    PromptReference,
    SuccessAuditData,
    OpenQuestionSourceOptionV1,
)
from app.services.redaction_service import RedactionService


ValidationLocationComponent = str | int


@dataclass(frozen=True, slots=True)
class AIValidationError:
    """Sanitized internal details for one structured-output validation error."""

    loc: tuple[ValidationLocationComponent, ...]
    type: str
    message: str


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

    def __init__(
        self,
        details: AIFailureDetails,
        *,
        retries_exhausted: bool = False,
        validation_errors: tuple[AIValidationError, ...] = (),
    ) -> None:
        self.details = details
        self._retries_exhausted = retries_exhausted
        self._validation_errors = validation_errors
        super().__init__(details.explanation)

    @property
    def retries_exhausted(self) -> bool:
        """Return whether the safe underlying failure exhausted its retry budget."""
        return self._retries_exhausted

    @property
    def validation_errors(self) -> tuple[AIValidationError, ...]:
        """Return sanitized internal structured-output validation diagnostics."""
        return self._validation_errors

    def __repr__(self) -> str:
        """Exclude provider responses and other internal audit fields."""
        return (
            f"{type(self).__name__}("
            f"category={self.details.category.value!r}, "
            f"request_identifier={self.details.request_identifier!r}, "
            f"explanation={self.details.explanation!r})"
        )


PromptResolver = Callable[[PromptReference], str]
PromptBundleValidator = Callable[[PromptBundle, AnalysisStage], None]


@dataclass(frozen=True, slots=True)
class StructuredResponseOutcome:
    """Internal result of provider-neutral response extraction and validation."""

    output: AIOutput | None
    failure_category: AIFailureCategory | None
    validation_errors: tuple[AIValidationError, ...] = ()

    def __post_init__(self) -> None:
        has_output = self.output is not None
        has_failure = self.failure_category is not None
        if has_output == has_failure:
            raise ValueError(
                "structured response outcome requires one output or failure"
            )
        has_validation_errors = bool(self.validation_errors)
        is_schema_failure = self.failure_category is AIFailureCategory.SCHEMA_VALIDATION
        if has_validation_errors != is_schema_failure:
            raise ValueError(
                "validation diagnostics require a schema-validation failure"
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
    OutputSchemaIdentifier.OPEN_QUESTIONS_V1: OpenQuestionsOutputV1,
    OutputSchemaIdentifier.POSTMORTEM_V1: PostmortemOutputV1,
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


def build_model_facing_output_schema(
    output_model: type[AIOutput],
    *,
    open_question_sources: tuple[OpenQuestionSourceOptionV1, ...] = (),
) -> dict[str, object]:
    """Return a schema copy that excludes application-owned generated metadata."""
    schema = deepcopy(output_model.model_json_schema())
    if output_model is HypothesesOutputV1:
        definitions = schema.get("$defs")
        if not isinstance(definitions, dict):
            raise AIProviderConfigurationError(
                "The hypotheses output schema is unavailable."
            )
        hypothesis_schema = definitions.get(HypothesisV1.__name__)
        if not isinstance(hypothesis_schema, dict):
            raise AIProviderConfigurationError(
                "The hypothesis item schema is unavailable."
            )
        properties = hypothesis_schema.get("properties")
        required = hypothesis_schema.get("required")
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise AIProviderConfigurationError(
                "The hypothesis metadata schema is unavailable."
            )

        hypothesis_schema["properties"] = {
            key: value for key, value in properties.items() if key != "hypothesis_id"
        }
        hypothesis_schema["required"] = [
            field_name for field_name in required if field_name != "hypothesis_id"
        ]
        return schema

    if output_model is not OpenQuestionsOutputV1:
        return schema
    if not open_question_sources:
        raise AIProviderConfigurationError(
            "The open-question source allowlist is unavailable."
        )

    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise AIProviderConfigurationError(
            "The open-question output schema is unavailable."
        )
    question_schema = definitions.get(OpenQuestionV1.__name__)
    if not isinstance(question_schema, dict):
        raise AIProviderConfigurationError(
            "The open-question item schema is unavailable."
        )
    properties = question_schema.get("properties")
    required = question_schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise AIProviderConfigurationError(
            "The open-question source schema is unavailable."
        )

    question_schema["properties"] = {
        **{
            key: value
            for key, value in properties.items()
            if key not in {"source_kind", "source_reference"}
        },
        "source_id": {
            "title": "Source Id",
            "type": "string",
            "enum": [source.source_id for source in open_question_sources],
        },
    }
    question_schema["required"] = [
        field_name
        for field_name in required
        if field_name not in {"source_kind", "source_reference"}
    ]
    question_schema["required"].append("source_id")
    return schema


def process_structured_response(
    raw_response: str | None,
    output_model: type[AIOutput],
    *,
    open_question_sources: tuple[OpenQuestionSourceOptionV1, ...] = (),
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
    response_data = _assign_application_owned_hypothesis_ids(
        response_data,
        output_model,
    )
    response_data = _restore_application_owned_open_question_sources(
        response_data,
        output_model,
        open_question_sources,
    )
    try:
        output = output_model.model_validate(response_data)
    except ValidationError as error:
        return StructuredResponseOutcome(
            output=None,
            failure_category=AIFailureCategory.SCHEMA_VALIDATION,
            validation_errors=_sanitize_validation_errors(error),
        )
    return StructuredResponseOutcome(
        output=output,
        failure_category=None,
    )


def _assign_application_owned_hypothesis_ids(
    response_data: object,
    output_model: type[AIOutput],
) -> object:
    if output_model is not HypothesesOutputV1 or not isinstance(response_data, dict):
        return response_data
    hypotheses = response_data.get("hypotheses")
    if not isinstance(hypotheses, list):
        return response_data

    normalized_hypotheses = []
    for index, hypothesis in enumerate(hypotheses, start=1):
        if not isinstance(hypothesis, dict):
            normalized_hypotheses.append(hypothesis)
            continue
        normalized_hypotheses.append(
            {
                **hypothesis,
                "hypothesis_id": f"H-{index:03d}",
            }
        )
    return {
        **response_data,
        "hypotheses": normalized_hypotheses,
    }


def _restore_application_owned_open_question_sources(
    response_data: object,
    output_model: type[AIOutput],
    source_options: tuple[OpenQuestionSourceOptionV1, ...],
) -> object:
    if (
        output_model is not OpenQuestionsOutputV1
        or not source_options
        or not isinstance(response_data, dict)
    ):
        return response_data
    questions = response_data.get("questions")
    if not isinstance(questions, list):
        return response_data

    sources_by_id = {source.source_id: source for source in source_options}
    sources_by_value = {
        (source.source_kind.value, source.source_reference): source
        for source in source_options
    }
    normalized_questions = []
    for question in questions:
        if not isinstance(question, dict):
            normalized_questions.append(question)
            continue

        source = None
        if "source_id" in question:
            source_id = question["source_id"]
            if isinstance(source_id, str):
                source = sources_by_id.get(source_id)
        else:
            source_kind = question.get("source_kind")
            source_reference = question.get("source_reference")
            if isinstance(source_kind, str) and isinstance(source_reference, str):
                source = sources_by_value.get(
                    (
                        source_kind,
                        source_reference,
                    )
                )
        if source is None:
            normalized_questions.append(question)
            continue

        normalized_question = {
            key: value
            for key, value in question.items()
            if key not in {"source_id", "source_kind", "source_reference"}
        }
        normalized_question["source_kind"] = source.source_kind.value
        normalized_question["source_reference"] = source.source_reference
        normalized_questions.append(normalized_question)
    return {
        **response_data,
        "questions": normalized_questions,
    }


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


def ai_result_matches_request(
    result: AIResult[AIOutput],
    *,
    request: AIRequest,
    output_type: type[BaseModel],
    provider_name: str,
    model_name: str,
) -> bool:
    """Check one provider result against its typed request traceability."""
    metadata = result.metadata
    return (
        isinstance(result.output, output_type)
        and metadata.analysis_stage is request.metadata.analysis_stage
        and metadata.output_schema is request.output_schema
        and metadata.system_prompt == request.prompts.system
        and metadata.task_prompt == request.prompts.task
        and metadata.request_identifier == request.metadata.request_identifier
        and metadata.provider_name == provider_name
        and metadata.model_name == model_name
    )


def raise_ai_provider_failure(
    *,
    request: AIRequest,
    category: AIFailureCategory,
    attempt_count: int,
    retries_exhausted: bool = False,
    validation_errors: tuple[AIValidationError, ...] = (),
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
        ),
        retries_exhausted=retries_exhausted,
        validation_errors=validation_errors,
    ) from None


def _sanitize_validation_errors(
    error: ValidationError,
) -> tuple[AIValidationError, ...]:
    return tuple(
        AIValidationError(
            loc=tuple(
                _sanitize_validation_location(component) for component in item["loc"]
            ),
            type=_sanitize_validation_type(item["type"]),
            message=_sanitize_validation_message(item["msg"]),
        )
        for item in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    )


def _sanitize_validation_location(
    component: object,
) -> ValidationLocationComponent:
    if isinstance(component, int) and not isinstance(component, bool):
        return component
    text = RedactionService.redact_text(str(component)).redacted_text
    return text if text.strip() else "[invalid-location]"


def _sanitize_validation_type(value: object) -> str:
    text = str(value)
    if re.fullmatch(r"[a-z][a-z0-9_.-]{0,99}", text) is None:
        return "validation_error"
    return text


def _sanitize_validation_message(value: object) -> str:
    text = RedactionService.redact_text(str(value)).redacted_text
    normalized = " ".join(text.split())
    return normalized[:500] if normalized else "Structured output failed validation."


def resolve_request_prompts(
    request: AIRequest,
    *,
    prompt_resolver: PromptResolver,
    prompt_bundle_validator: PromptBundleValidator,
) -> tuple[str, str]:
    """Validate a request's prompt mapping and return both registered contents."""
    # noinspection PyBroadException
    try:
        prompt_bundle_validator(
            request.prompts,
            request.metadata.analysis_stage,
        )
        system_prompt = prompt_resolver(request.prompts.system)
        task_prompt = prompt_resolver(request.prompts.task)
    except (LookupError, OSError, UnicodeError, ValueError):
        raise_ai_provider_failure(
            request=request,
            category=AIFailureCategory.UNKNOWN_PROMPT,
            attempt_count=1,
        )
    except Exception:
        # Both callables are injected boundaries; sanitize undocumented failures.
        raise_ai_provider_failure(
            request=request,
            category=AIFailureCategory.UNKNOWN_PROMPT,
            attempt_count=1,
        )
    if (
        not isinstance(system_prompt, str)
        or not isinstance(task_prompt, str)
        or not system_prompt.strip()
        or not task_prompt.strip()
    ):
        raise_ai_provider_failure(
            request=request,
            category=AIFailureCategory.UNKNOWN_PROMPT,
            attempt_count=1,
        )
    return system_prompt, task_prompt


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
