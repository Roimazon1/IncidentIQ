"""Environment-backed application configuration."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables and a local dotenv file."""

    app_name: str = "IncidentIQ"
    database_url: str = "sqlite:///./incidentiq.db"
    ai_provider: str = "fake"
    openai_api_key: SecretStr | None = None
    max_upload_bytes: int = Field(default=10_485_760, gt=0)
    debug: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""

    return Settings()
