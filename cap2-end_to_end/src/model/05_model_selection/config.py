"""
Configuration module for model selection
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def find_env_file() -> Path:
    """Find .env file by searching up from current directory."""
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        env_file = parent / ".env"
        if env_file.exists():
            return env_file
    return Path(".env")


class ComponentSettings(BaseSettings):
    """Settings for model selection component."""

    model_config = SettingsConfigDict(
        env_file=str(find_env_file()), env_file_encoding="utf-8", extra="ignore"
    )

    GCS_BUCKET_NAME: str = ""
    GCP_PROJECT_ID: str = ""
    GCP_REGION: str = "us-central1"
    WANDB_PROJECT: str = "housing-mlops-gcp"
    WANDB_ENTITY: str | None = None


settings = ComponentSettings()
