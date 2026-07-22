"""Settings-driven construction for the Phase 6 analysis service facade."""

from pathlib import Path

from sqlalchemy.orm import Session

from app.config import Settings
from app.services.ai_provider import AIProvider, AIProviderFactory
from app.services.analysis_service import AnalysisService
from app.services.analysis_stage_runner import AnalysisProviderRequiredError
from app.services.prompt_registry import PromptRegistry
from app.services.providers.fake_provider import FakeAIProvider
from app.services.providers.gemini_provider import GeminiAIProvider


_FAKE_RESPONSE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "fake_ai_core_responses.json"
)
_CORE_FAKE_FIXTURES = (
    "valid_summary",
    "valid_timeline",
    "valid_hypotheses",
    "valid_critic",
)


def build_configured_analysis_service(
    session: Session,
    settings: Settings,
) -> AnalysisService:
    """Build an analysis service with the settings-selected concrete provider."""
    prompt_registry = PromptRegistry()

    def build_fake_provider(configured_settings: Settings) -> AIProvider:
        del configured_settings
        return FakeAIProvider.from_file_set(
            _FAKE_RESPONSE_FIXTURE_PATH,
            _CORE_FAKE_FIXTURES,
            prompt_resolver=prompt_registry.resolve_content,
            prompt_bundle_validator=prompt_registry.validate_bundle,
        )

    def build_gemini_provider(configured_settings: Settings) -> AIProvider:
        return GeminiAIProvider.from_settings(
            configured_settings,
            prompt_resolver=prompt_registry.resolve_content,
            prompt_bundle_validator=prompt_registry.validate_bundle,
        )

    provider = AIProviderFactory(
        fake_builder=build_fake_provider,
        gemini_builder=build_gemini_provider,
    ).create(settings)
    model_name = (
        FakeAIProvider.model_name
        if settings.ai_provider == FakeAIProvider.provider_name
        else settings.gemini_model
    )
    if model_name is None:
        raise AnalysisProviderRequiredError(
            "A configured AI provider model is required to start analysis."
        )
    return AnalysisService(
        session,
        ai_provider=provider,
        configured_provider_name=settings.ai_provider,
        configured_model_name=model_name,
    )
