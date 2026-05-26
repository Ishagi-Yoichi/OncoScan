from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    app_name: str = "OncoScan AI"
    environment: str = "development"
    model_path: Path | None = None
    device: str = "cpu"
    max_upload_mb: int = 10
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ONCOSCAN_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
