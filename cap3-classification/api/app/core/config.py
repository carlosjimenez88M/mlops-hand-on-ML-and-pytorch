"""API settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the inference API."""

    project_name: str = "cap3-classification-api"
    version: str = "0.1.0"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_model_name: str = "chapter3_digit_classifier"
    mlflow_model_alias: str = "staging"
    local_model_path: str = "models/registered/latest_model.joblib"
    serving_metadata_path: str = "artifacts/serving/model_info.json"

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )
