"""Environment-backed application configuration."""

from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables and a local dotenv file."""

    app_name: str = "IncidentIQ"
    database_url: str = "sqlite:///./incidentiq.db"
    ai_provider: str = "fake"
    gemini_api_key: SecretStr | None = None
    gemini_model: str | None = None
    max_upload_bytes: int = Field(default=10_485_760, gt=0)
    display_timezone: str = "UTC"
    debug: bool = True

    @field_validator("display_timezone")
    @classmethod
    def validate_display_timezone(cls, value: str) -> str:
        """Require an IANA timezone name supported by the runtime."""
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("display_timezone must be a valid IANA timezone") from exc
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""

    return Settings()
