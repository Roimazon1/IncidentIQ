"""Focused tests for the provider protocol and settings-driven factory."""

from __future__ import annotations

from inspect import signature
from pathlib import Path

import pytest

from app.config import Settings
from app.schemas.ai_outputs import AIOutput
from app.schemas.ai_provider import AIFailureCategory, AIRequest, AIResult
from app.services.ai_provider import (
    AIProvider,
    AIProviderConfigurationError,
    AIProviderFactory,
)


class _StubProvider:
    def generate(self, request: AIRequest) -> AIResult[AIOutput]:
        raise AssertionError("selection tests must not invoke the provider")


def _settings(
    *,
    ai_provider: str,
    gemini_api_key: str | None = None,
    gemini_model: str | None = None,
) -> Settings:
    return Settings(
        _env_file=None,
        ai_provider=ai_provider,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
    )


def test_ai_provider_interface_accepts_only_a_typed_request() -> None:
    parameters = signature(AIProvider.generate).parameters

    assert list(parameters) == ["self", "request"]
    assert isinstance(_StubProvider(), AIProvider)


def test_provider_factory_selects_fake_without_gemini_credentials() -> None:
    provider = _StubProvider()
    received_settings: list[Settings] = []

    def build_fake(settings: Settings) -> AIProvider:
        received_settings.append(settings)
        return provider

    settings = _settings(ai_provider="fake")
    factory = AIProviderFactory(fake_builder=build_fake)

    selected = factory.create(settings)

    assert selected is provider
    assert received_settings == [settings]


def test_provider_factory_selects_gemini_with_complete_configuration() -> None:
    provider = _StubProvider()
    received_settings: list[Settings] = []

    def build_gemini(settings: Settings) -> AIProvider:
        received_settings.append(settings)
        return provider

    settings = _settings(
        ai_provider="gemini",
        gemini_api_key="test-gemini-key",
        gemini_model="gemini-2.5-flash",
    )
    factory = AIProviderFactory(gemini_builder=build_gemini)

    selected = factory.create(settings)

    assert selected is provider
    assert received_settings == [settings]


@pytest.mark.parametrize(
    ("api_key", "model", "missing_names"),
    [
        (None, "gemini-2.5-flash", "GEMINI_API_KEY"),
        ("test-gemini-key", None, "GEMINI_MODEL"),
        ("", "", "GEMINI_API_KEY and GEMINI_MODEL"),
    ],
)
def test_gemini_selection_fails_before_builder_when_configuration_is_missing(
    api_key: str | None,
    model: str | None,
    missing_names: str,
) -> None:
    builder_called = False

    def build_gemini(settings: Settings) -> AIProvider:
        nonlocal builder_called
        builder_called = True
        return _StubProvider()

    factory = AIProviderFactory(gemini_builder=build_gemini)

    with pytest.raises(AIProviderConfigurationError) as error_info:
        factory.create(
            _settings(
                ai_provider="gemini",
                gemini_api_key=api_key,
                gemini_model=model,
            )
        )

    error = error_info.value
    assert builder_called is False
    assert error.details.category is AIFailureCategory.CONFIGURATION
    assert missing_names in str(error)
    assert "test-gemini-key" not in str(error)
    assert "test-gemini-key" not in repr(error)


def test_provider_factory_rejects_unknown_provider_without_echoing_value() -> None:
    unsafe_value = "unknown\nforged-log-entry"
    factory = AIProviderFactory(fake_builder=lambda settings: _StubProvider())

    with pytest.raises(AIProviderConfigurationError) as error_info:
        factory.create(_settings(ai_provider=unsafe_value))

    error = error_info.value
    assert error.details.category is AIFailureCategory.CONFIGURATION
    assert error.__cause__ is None
    assert unsafe_value not in str(error)
    assert unsafe_value not in repr(error)
    assert "fake" in str(error)
    assert "gemini" in str(error)


def test_provider_factory_rejects_unregistered_selected_implementation() -> None:
    factory = AIProviderFactory()

    with pytest.raises(
        AIProviderConfigurationError,
        match="implementation is not registered",
    ):
        factory.create(_settings(ai_provider="fake"))


def test_provider_factory_rejects_builder_result_outside_protocol() -> None:
    factory = AIProviderFactory(fake_builder=lambda settings: object())

    with pytest.raises(
        AIProviderConfigurationError,
        match="implementation is invalid",
    ):
        factory.create(_settings(ai_provider="fake"))


def test_routers_do_not_access_ai_provider_boundary_directly() -> None:
    forbidden_terms = {
        "AIProvider",
        "AIProviderFactory",
        "AIRequest",
        "GeminiAIProvider",
        "app.schemas.ai_provider",
        "app.services.ai_provider",
        "google.genai",
    }

    for router_path in Path("app/routers").glob("*.py"):
        router_source = router_path.read_text(encoding="utf-8")
        assert all(term not in router_source for term in forbidden_terms), router_path
