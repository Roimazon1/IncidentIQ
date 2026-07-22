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

from app.schemas.ai_outputs import AIOutput
from app.schemas.ai_provider import (
    AIFailureCategory,
    AIRequest,
    AIResult,
)
from app.services.ai_provider import (
    AIProviderConfigurationError,
    build_ai_result,
    process_structured_response,
    raise_ai_provider_failure,
    select_output_model,
)

_SIMULATED_FAILURE_CATEGORIES = frozenset(
    {
        AIFailureCategory.TRANSIENT_PROVIDER_FAILURE,
        AIFailureCategory.AUTHENTICATION,
    }
)


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
        if has_failure and self.failure_category not in _SIMULATED_FAILURE_CATEGORIES:
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
        output_model = select_output_model(request)
        fixture = self._fixture
        if fixture.failure_category is not None:
            raise_ai_provider_failure(
                request=request,
                category=fixture.failure_category,
                attempt_count=1,
            )

        raw_response = fixture.raw_response
        if raw_response is None or fixture.output_schema != request.output_schema.value:
            raise_ai_provider_failure(
                request=request,
                category=AIFailureCategory.UNSUPPORTED_OUTPUT_SCHEMA,
                attempt_count=1,
                raw_response=raw_response,
            )

        outcome = process_structured_response(raw_response, output_model)
        if outcome.failure_category is not None:
            raise_ai_provider_failure(
                request=request,
                category=outcome.failure_category,
                attempt_count=1,
                raw_response=raw_response,
            )

        output = outcome.output
        if output is None:
            raise AssertionError("validated fake response did not contain output")
        return build_ai_result(
            request=request,
            output=output,
            provider_name=self.provider_name,
            model_name=self.model_name,
            attempt_count=1,
            raw_response=raw_response,
        )
