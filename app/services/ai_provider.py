"""Provider-neutral AI interface and settings-driven provider selection."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Protocol, runtime_checkable

from app.config import Settings
from app.schemas.ai_outputs import AIOutput
from app.schemas.ai_provider import (
    AIFailureCategory,
    AIFailureDetails,
    AIRequest,
    AIResult,
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
