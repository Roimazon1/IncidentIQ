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
    PromptBundleValidator,
    PromptResolver,
    build_ai_result,
    process_structured_response,
    raise_ai_provider_failure,
    resolve_request_prompts,
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
    """Return configured fixtures through the normal typed provider boundary."""

    provider_name = "fake"
    model_name = "fixture-v1"

    def __init__(
        self,
        fixtures: tuple[_FakeResponseFixture, ...],
        *,
        prompt_resolver: PromptResolver,
        prompt_bundle_validator: PromptBundleValidator,
    ) -> None:
        self._fixtures = fixtures
        self._prompt_resolver = prompt_resolver
        self._prompt_bundle_validator = prompt_bundle_validator

    @classmethod
    def from_file(
        cls,
        path: Path,
        fixture_name: str,
        *,
        prompt_resolver: PromptResolver,
        prompt_bundle_validator: PromptBundleValidator,
    ) -> Self:
        """Load and validate a named fixture without environment or network access."""
        fixtures = cls._load_fixture_bank(path)
        fixture = fixtures.get(fixture_name)
        if fixture is None:
            raise AIProviderConfigurationError(
                "The requested fake AI response fixture is not registered."
            )
        return cls(
            (fixture,),
            prompt_resolver=prompt_resolver,
            prompt_bundle_validator=prompt_bundle_validator,
        )

    @classmethod
    def from_file_set(
        cls,
        path: Path,
        fixture_names: tuple[str, ...],
        *,
        prompt_resolver: PromptResolver,
        prompt_bundle_validator: PromptBundleValidator,
    ) -> Self:
        """Load one deterministic response fixture for each requested schema."""
        fixture_bank = cls._load_fixture_bank(path)
        selected_fixtures = tuple(
            fixture_bank.get(fixture_name) for fixture_name in fixture_names
        )
        if not selected_fixtures or any(
            fixture is None for fixture in selected_fixtures
        ):
            raise AIProviderConfigurationError(
                "A requested fake AI response fixture is not registered."
            )

        typed_fixtures = tuple(
            fixture for fixture in selected_fixtures if fixture is not None
        )
        output_schemas = [fixture.output_schema for fixture in typed_fixtures]
        if any(output_schema is None for output_schema in output_schemas) or len(
            output_schemas
        ) != len(set(output_schemas)):
            raise AIProviderConfigurationError(
                "The fake AI response fixture set is invalid."
            )
        return cls(
            typed_fixtures,
            prompt_resolver=prompt_resolver,
            prompt_bundle_validator=prompt_bundle_validator,
        )

    @staticmethod
    def _load_fixture_bank(path: Path) -> dict[str, _FakeResponseFixture]:
        try:
            fixture_document = json.loads(path.read_text(encoding="utf-8"))
            return _FIXTURE_BANK_ADAPTER.validate_python(fixture_document)
        except (OSError, json.JSONDecodeError, ValidationError):
            raise AIProviderConfigurationError(
                "The fake AI response fixture file is invalid."
            ) from None

    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        """Parse and validate the configured fixture as the requested output type."""
        output_model = select_output_model(request)
        resolve_request_prompts(
            request,
            prompt_resolver=self._prompt_resolver,
            prompt_bundle_validator=self._prompt_bundle_validator,
        )
        fixture = self._select_fixture(request)
        if fixture is None:
            raise_ai_provider_failure(
                request=request,
                category=AIFailureCategory.UNSUPPORTED_OUTPUT_SCHEMA,
                attempt_count=1,
            )
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

    def _select_fixture(self, request: AIRequest) -> _FakeResponseFixture | None:
        if len(self._fixtures) == 1:
            return self._fixtures[0]
        return next(
            (
                fixture
                for fixture in self._fixtures
                if fixture.output_schema == request.output_schema.value
            ),
            None,
        )
