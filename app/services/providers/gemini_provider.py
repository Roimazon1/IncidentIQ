"""Gemini Developer API adapter isolated behind provider-neutral contracts."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict

from pydantic import TypeAdapter, ValidationError

from app.config import Settings
from app.schemas.ai_outputs import AIOutput
from app.schemas.ai_provider import (
    AIFailureCategory,
    AIRequest,
    AIResult,
    LogSafeName,
)
from app.services.ai_provider import (
    AIProviderConfigurationError,
    BoundedRetryPolicy,
    PromptBundleValidator,
    PromptResolver,
    build_ai_result,
    process_structured_response,
    raise_ai_provider_failure,
    resolve_request_prompts,
    select_output_model,
)

if TYPE_CHECKING:
    # noinspection PyPackageRequirements
    from google.genai.client import Client as _OfficialClient

    # noinspection PyPackageRequirements
    from google.genai.types import GenerateContentResponse as _OfficialResponse


class GeminiGenerateConfig(TypedDict):
    """Narrow provider-local configuration passed to Gemini models."""

    system_instruction: str
    response_mime_type: str
    response_json_schema: dict[str, object]


class GeminiResponseProtocol(ABC):
    """Response surface required by the concrete Gemini provider."""

    @property
    @abstractmethod
    def text(self) -> str | None: ...


class GeminiModelsProtocol(ABC):
    """Model-generation surface required by the concrete Gemini provider."""

    @abstractmethod
    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: GeminiGenerateConfig,
    ) -> GeminiResponseProtocol: ...


class GeminiClientProtocol(ABC):
    """Client surface required by the concrete Gemini provider."""

    @property
    @abstractmethod
    def models(self) -> GeminiModelsProtocol: ...


class _OfficialGeminiResponseAdapter(GeminiResponseProtocol):
    """Expose only response text from the official SDK response."""

    def __init__(self, response: _OfficialResponse) -> None:
        self._response = response

    @property
    def text(self) -> str | None:
        return self._response.text


class _OfficialGeminiModelsAdapter(GeminiModelsProtocol):
    """Delegate the narrow generation call to an official SDK client."""

    def __init__(self, client: _OfficialClient) -> None:
        self._client = client

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: GeminiGenerateConfig,
    ) -> GeminiResponseProtocol:
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        return _OfficialGeminiResponseAdapter(response)


class _OfficialGeminiClientAdapter(GeminiClientProtocol):
    """Adapt the official client to the provider's narrow protocol."""

    def __init__(self, client: _OfficialClient) -> None:
        self._models = _OfficialGeminiModelsAdapter(client)

    @property
    def models(self) -> GeminiModelsProtocol:
        return self._models


Sleeper = Callable[[float], None]

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 1.0

_GEMINI_SCHEMA_KEYS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "anyOf",
        "description",
        "enum",
        "format",
        "items",
        "maxItems",
        "maximum",
        "minItems",
        "minimum",
        "oneOf",
        "prefixItems",
        "properties",
        "required",
        "title",
        "type",
    }
)
_NAMED_SCHEMA_MAP_KEYS = frozenset({"$defs", "properties"})
_MODEL_NAME_ADAPTER = TypeAdapter(LogSafeName)


class _UnavailableGeminiAPIError(Exception):
    """Fallback type used only when the optional runtime SDK is not installed."""


def _load_official_api_error_type() -> type[Exception]:
    try:
        # noinspection PyPackageRequirements
        from google.genai.errors import APIError
    except ImportError:
        return _UnavailableGeminiAPIError
    return APIError


_OFFICIAL_API_ERROR_TYPE = _load_official_api_error_type()


def _gemini_supported_schema(schema: dict[str, object]) -> dict[str, object]:
    """Reduce Pydantic JSON Schema to Gemini's documented supported subset."""

    def sanitize(value: object, *, named_schema_map: bool = False) -> object:
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if not isinstance(value, dict):
            return value
        if named_schema_map:
            return {name: sanitize(definition) for name, definition in value.items()}

        return {
            key: sanitize(item, named_schema_map=key in _NAMED_SCHEMA_MAP_KEYS)
            for key, item in value.items()
            if key in _GEMINI_SCHEMA_KEYS
        }

    sanitized_schema = sanitize(schema)
    if not isinstance(sanitized_schema, dict):
        raise TypeError("Gemini response schema must remain an object schema.")
    return sanitized_schema


class GeminiAIProvider:
    """Translate typed redacted requests into bounded Gemini model calls."""

    provider_name = "gemini"

    def __init__(
        self,
        *,
        model_name: str | None,
        prompt_resolver: PromptResolver,
        prompt_bundle_validator: PromptBundleValidator,
        client: GeminiClientProtocol | None = None,
        api_key: str | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        sleeper: Sleeper = time.sleep,
        api_error_type: type[Exception] = _OFFICIAL_API_ERROR_TYPE,
    ) -> None:
        self.model_name = self._validate_model_name(model_name)
        self._prompt_resolver = prompt_resolver
        self._prompt_bundle_validator = prompt_bundle_validator
        self._retry_policy = BoundedRetryPolicy(
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        self._sleeper = sleeper
        self._api_error_type = api_error_type
        self._client = (
            client if client is not None else self._create_real_client(api_key)
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        prompt_resolver: PromptResolver,
        prompt_bundle_validator: PromptBundleValidator,
        client: GeminiClientProtocol | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        sleeper: Sleeper = time.sleep,
        api_error_type: type[Exception] = _OFFICIAL_API_ERROR_TYPE,
    ) -> GeminiAIProvider:
        """Construct from application settings without reading secrets for test clients."""
        api_key = None
        if client is None and settings.gemini_api_key is not None:
            api_key = settings.gemini_api_key.get_secret_value()
        return cls(
            model_name=settings.gemini_model,
            prompt_resolver=prompt_resolver,
            prompt_bundle_validator=prompt_bundle_validator,
            client=client,
            api_key=api_key,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            sleeper=sleeper,
            api_error_type=api_error_type,
        )

    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        """Call Gemini with redacted data and return only locally validated output."""
        output_model = select_output_model(request)
        system_prompt, task_prompt = resolve_request_prompts(
            request,
            prompt_resolver=self._prompt_resolver,
            prompt_bundle_validator=self._prompt_bundle_validator,
        )
        contents = self._build_contents(request, task_prompt)
        config: GeminiGenerateConfig = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_json_schema": _gemini_supported_schema(
                output_model.model_json_schema()
            ),
        }

        for attempt_count in self._retry_policy.attempt_numbers:
            # noinspection PyBroadException
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
            except TimeoutError:
                self._handle_provider_failure(
                    request=request,
                    category=AIFailureCategory.TIMEOUT,
                    retryable=True,
                    attempt_count=attempt_count,
                )
                continue
            except self._api_error_type as api_error:
                category, retryable = self._classify_api_error(api_error)
                self._handle_provider_failure(
                    request=request,
                    category=category,
                    retryable=retryable,
                    attempt_count=attempt_count,
                )
                continue
            except Exception:
                # Third-party and injected clients can raise undocumented exceptions.
                # Sanitize them at this boundary and never retry them.
                raise_ai_provider_failure(
                    request=request,
                    category=AIFailureCategory.TRANSIENT_PROVIDER_FAILURE,
                    attempt_count=attempt_count,
                )

            # noinspection PyBroadException
            try:
                raw_response = response.text
            except Exception:
                # Response properties are third-party code and can fail unexpectedly.
                # Unknown extraction failures are sanitized without another attempt.
                raise_ai_provider_failure(
                    request=request,
                    category=AIFailureCategory.TRANSIENT_PROVIDER_FAILURE,
                    attempt_count=attempt_count,
                )
            outcome = process_structured_response(
                raw_response,
                output_model,
            )
            if outcome.failure_category is not None:
                if self._retry_policy.has_next_attempt(attempt_count):
                    self._sleep_before_retry(attempt_count)
                    continue
                raise_ai_provider_failure(
                    request=request,
                    category=AIFailureCategory.EXHAUSTED_RETRIES,
                    attempt_count=attempt_count,
                    raw_response=raw_response,
                )

            output = outcome.output
            if output is None or raw_response is None:
                raise AssertionError("validated Gemini response did not contain output")
            return build_ai_result(
                request=request,
                output=output,
                provider_name=self.provider_name,
                model_name=self.model_name,
                attempt_count=attempt_count,
                raw_response=raw_response,
            )

        raise AssertionError("bounded Gemini attempt loop did not return or raise")

    @staticmethod
    def _validate_model_name(model_name: str | None) -> str:
        try:
            return _MODEL_NAME_ADAPTER.validate_python(model_name)
        except ValidationError:
            raise AIProviderConfigurationError(
                "Gemini provider configuration requires a valid GEMINI_MODEL."
            ) from None

    @staticmethod
    def _create_real_client(api_key: str | None) -> GeminiClientProtocol:
        if api_key is None or not api_key.strip():
            raise AIProviderConfigurationError(
                "Gemini provider configuration requires GEMINI_API_KEY."
            )
        try:
            # noinspection PyPackageRequirements
            from google import genai
        except ImportError:
            raise AIProviderConfigurationError(
                "The Gemini provider dependency is unavailable."
            ) from None
        # noinspection PyBroadException
        try:
            return _OfficialGeminiClientAdapter(genai.Client(api_key=api_key))
        except (TypeError, ValueError):
            raise AIProviderConfigurationError(
                "The Gemini provider client could not be constructed."
            ) from None
        except Exception:
            # Sanitize undocumented SDK construction failures without exposing a key.
            raise AIProviderConfigurationError(
                "The Gemini provider client could not be constructed."
            ) from None

    @staticmethod
    def _build_contents(request: AIRequest, task_prompt: str) -> str:
        payload: dict[str, object] = {
            "task_prompt": task_prompt,
            "output_schema": request.output_schema.value,
            "metadata": {
                "request_identifier": request.metadata.request_identifier,
                "incident_public_identifier": (
                    request.metadata.incident_public_identifier
                ),
                "analysis_stage": request.metadata.analysis_stage.value,
                "evidence_manifest_checksum": (
                    request.metadata.evidence_manifest_checksum
                ),
            },
        }
        if request.evidence_manifest is not None:
            payload["evidence_manifest"] = request.evidence_manifest.model_dump(
                mode="json"
            )
        if request.report_input is not None:
            payload["report_input"] = request.report_input.model_dump(mode="json")
        if request.critic_context is not None:
            payload["critic_context"] = request.critic_context.model_dump(mode="json")
        if request.bias_context is not None:
            payload["bias_context"] = request.bias_context.model_dump(mode="json")
        if request.open_questions_context is not None:
            payload["open_questions_context"] = (
                request.open_questions_context.model_dump(mode="json")
            )
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _classify_api_error(
        api_error: Exception,
    ) -> tuple[AIFailureCategory, bool]:
        # noinspection PyBroadException
        try:
            status_code = getattr(api_error, "code", None)
        except Exception:
            # An SDK error with an unreadable code is not positively recoverable.
            return AIFailureCategory.TRANSIENT_PROVIDER_FAILURE, False
        if isinstance(status_code, bool) or not isinstance(status_code, int):
            return AIFailureCategory.TRANSIENT_PROVIDER_FAILURE, False
        if status_code == 429:
            return AIFailureCategory.RATE_LIMIT, True
        if status_code in {401, 403}:
            return AIFailureCategory.AUTHENTICATION, False
        if isinstance(status_code, int) and 500 <= status_code <= 599:
            return AIFailureCategory.TRANSIENT_PROVIDER_FAILURE, True
        if isinstance(status_code, int) and 400 <= status_code <= 499:
            return AIFailureCategory.UNSUPPORTED_OUTPUT_SCHEMA, False
        return AIFailureCategory.TRANSIENT_PROVIDER_FAILURE, False

    def _handle_provider_failure(
        self,
        *,
        request: AIRequest,
        category: AIFailureCategory,
        retryable: bool,
        attempt_count: int,
    ) -> None:
        if retryable and self._retry_policy.has_next_attempt(attempt_count):
            self._sleep_before_retry(attempt_count)
            return
        final_category = AIFailureCategory.EXHAUSTED_RETRIES if retryable else category
        raise_ai_provider_failure(
            request=request,
            category=final_category,
            attempt_count=attempt_count,
        )

    def _sleep_before_retry(self, attempt_count: int) -> None:
        delay = self._retry_policy.delay_before_next_attempt(attempt_count)
        self._sleeper(delay)
