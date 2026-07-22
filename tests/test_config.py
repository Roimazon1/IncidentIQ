"""Tests for environment-backed application configuration."""

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def test_defaults_and_local_dotenv_values_load(tmp_path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "APP_NAME=Local IncidentIQ\n"
        "DATABASE_URL=sqlite:///./local.db\n"
        "AI_PROVIDER=fake\n"
        "GEMINI_API_KEY=\n"
        "GEMINI_MODEL=\n"
        "MAX_UPLOAD_BYTES=2048\n"
        "DISPLAY_TIMEZONE=Asia/Jerusalem\n"
        "DEBUG=false\n"
        "UNKNOWN_SETTING=ignored\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=dotenv_path)

    assert settings.app_name == "Local IncidentIQ"
    assert settings.database_url == "sqlite:///./local.db"
    assert settings.ai_provider == "fake"
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == ""
    assert settings.gemini_model == ""
    assert settings.max_upload_bytes == 2048
    assert settings.display_timezone == "Asia/Jerusalem"
    assert settings.debug is False


def test_boolean_and_integer_environment_values_are_parsed(monkeypatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "4096")
    monkeypatch.setenv("DEBUG", "true")

    settings = Settings(_env_file=None)

    assert settings.max_upload_bytes == 4096
    assert settings.debug is True


def test_environment_variables_override_defaults(monkeypatch) -> None:
    monkeypatch.setenv("APP_NAME", "Environment IncidentIQ")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./environment.db")
    monkeypatch.setenv("AI_PROVIDER", "custom")
    monkeypatch.setenv("DISPLAY_TIMEZONE", "America/New_York")

    settings = Settings(_env_file=None)

    assert settings.app_name == "Environment IncidentIQ"
    assert settings.database_url == "sqlite:///./environment.db"
    assert settings.ai_provider == "custom"
    assert settings.display_timezone == "America/New_York"


def test_fake_provider_accepts_empty_gemini_settings(monkeypatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "fake")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "")

    settings = Settings(_env_file=None)

    assert settings.ai_provider == "fake"
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == ""
    assert settings.gemini_model == ""


def test_invalid_max_upload_bytes_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_display_timezone_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("DISPLAY_TIMEZONE", "Not/A_Timezone")

    with pytest.raises(ValidationError, match="valid IANA timezone"):
        Settings(_env_file=None)


def test_get_settings_returns_a_cached_instance(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    first_settings = get_settings()
    second_settings = get_settings()

    assert first_settings is second_settings
    get_settings.cache_clear()
