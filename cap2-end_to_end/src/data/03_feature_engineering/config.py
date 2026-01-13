"""
Configuration for feature engineering module using pydantic_settings
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


def find_env_file() -> Path:
    """Find .env file by searching up from current directory."""
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        env_file = parent / ".env"
        if env_file.exists():
            return env_file
    return Path(".env")  # Fallback


class ComponentSettings(BaseSettings):
    """Configuration for the feature engineering component."""

    model_config = SettingsConfigDict(
        env_file=str(find_env_file()),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    ###############################
    # GCP Components Settings
    ###############################
    GCS_BUCKET_NAME: str
    GCP_PROJECT_ID: Optional[str] = None
    GCP_REGION: str = "us-central1"
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None

    ###############################
    # Weights & Biases
    ###############################
    WANDB_PROJECT: str = "housing-mlops-gcp"
    WANDB_ENTITY: Optional[str] = None
    WANDB_API_KEY: Optional[str] = None

    @property
    def gcs_bucket_uri(self) -> str:
        """URI completa del bucket"""
        return f"gs://{self.GCS_BUCKET_NAME}"


settings = ComponentSettings()
