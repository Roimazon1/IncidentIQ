"""Deterministic fixture-backed provider for offline development and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    model_validator,
)

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
from app.services.ai_provider import (
    AIProviderConfigurationError,
    AIProviderExecutionError,
)


_OUTPUT_MODELS: dict[OutputSchemaIdentifier, type[BaseModel]] = {
    OutputSchemaIdentifier.SUMMARY_V1: SummaryOutputV1,
    OutputSchemaIdentifier.TIMELINE_V1: TimelineOutputV1,
    OutputSchemaIdentifier.HYPOTHESES_V1: HypothesesOutputV1,
    OutputSchemaIdentifier.CRITIC_V1: CriticOutputV1,
    OutputSchemaIdentifier.REASONING_RISKS_V1: ReasoningRisksOutputV1,
}

_SIMULATED_FAILURE_EXPLANATIONS = {
    AIFailureCategory.TRANSIENT_PROVIDER_FAILURE: (
        "The AI provider temporarily failed."
    ),
    AIFailureCategory.AUTHENTICATION: "The AI provider rejected its credentials.",
}


class _FakeResponseFixture(BaseModel):
    """Validated internal description of one deterministic fake outcome."""

    output_schema: str | None = None
    raw_response: str | None = None
    failure_category: AIFailureCategory | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def require_exactly_one_outcome(self) -> _FakeResponseFixture:
        has_response = self.output_schema is not None and self.raw_response is not None
        has_failure = self.failure_category is not None
        if has_response == has_failure:
            raise ValueError("fixture must define exactly one response or failure")
        if has_failure and self.failure_category not in _SIMULATED_FAILURE_EXPLANATIONS:
            raise ValueError("fixture contains an unsupported simulated failure")
        return self


_FIXTURE_BANK_ADAPTER = TypeAdapter(dict[str, _FakeResponseFixture])


class FakeAIProvider:
    """Return one named fixture through the normal typed provider boundary."""

    provider_name = "fake"
    model_name = "fixture-v1"

    def __init__(self, fixture: _FakeResponseFixture) -> None:
        self._fixture = fixture

    @classmethod
    def from_file(cls, path: Path, fixture_name: str) -> Self:
        """Load and validate a named fixture without environment or network access."""
        try:
            fixture_document = json.loads(path.read_text(encoding="utf-8"))
            fixtures = _FIXTURE_BANK_ADAPTER.validate_python(fixture_document)
        except (OSError, json.JSONDecodeError, ValidationError):
            raise AIProviderConfigurationError(
                "The fake AI response fixture file is invalid."
            ) from None

        fixture = fixtures.get(fixture_name)
        if fixture is None:
            raise AIProviderConfigurationError(
                "The requested fake AI response fixture is not registered."
            )
        return cls(fixture)

    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        """Parse and validate the configured fixture as the requested output type."""
        fixture = self._fixture
        if fixture.failure_category is not None:
            self._raise_failure(
                request=request,
                category=fixture.failure_category,
                explanation=_SIMULATED_FAILURE_EXPLANATIONS[fixture.failure_category],
            )

        raw_response = fixture.raw_response
        if raw_response is None or fixture.output_schema != request.output_schema.value:
            self._raise_failure(
                request=request,
                category=AIFailureCategory.UNSUPPORTED_OUTPUT_SCHEMA,
                explanation="The fake response does not support the requested schema.",
                raw_response=raw_response,
            )

        try:
            response_data = json.loads(raw_response)
        except json.JSONDecodeError:
            self._raise_failure(
                request=request,
                category=AIFailureCategory.MALFORMED_JSON,
                explanation="The AI provider returned malformed JSON.",
                raw_response=raw_response,
            )

        output_model = _OUTPUT_MODELS[request.output_schema]
        try:
            output = output_model.model_validate(response_data)
        except ValidationError:
            self._raise_failure(
                request=request,
                category=AIFailureCategory.SCHEMA_VALIDATION,
                explanation="The AI provider response failed schema validation.",
                raw_response=raw_response,
            )

        return AIResult[AIOutput](
            output=output,
            metadata=AIResultMetadata(
                provider_name=self.provider_name,
                model_name=self.model_name,
                system_prompt=request.prompts.system,
                task_prompt=request.prompts.task,
                analysis_stage=request.metadata.analysis_stage,
                output_schema=request.output_schema,
                request_identifier=request.metadata.request_identifier,
                attempt_count=1,
            ),
            audit=SuccessAuditData(raw_response=raw_response),
        )

    @staticmethod
    def _raise_failure(
        *,
        request: AIRequest,
        category: AIFailureCategory,
        explanation: str,
        raw_response: str | None = None,
    ) -> None:
        audit = FailureAuditData(
            request_identifier=request.metadata.request_identifier,
            attempt_count=1,
            raw_response=raw_response,
        )
        raise AIProviderExecutionError(
            AIFailureDetails(
                category=category,
                request_identifier=request.metadata.request_identifier,
                explanation=explanation,
                audit=audit,
            )
        )
