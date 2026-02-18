"""Environment-backed configuration for registration step (optional helper)."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RegistrationConfig(BaseSettings):
    """Configuration for model registration."""

    bucket_name: str = Field(..., description="GCS bucket name")
    gcs_train_path: str = Field(..., description="Path to training data in GCS")
    gcs_test_path: str = Field(..., description="Path to test data in GCS")

    registered_model_name: str = Field(
        default="housing_price_model", description="Registered model name"
    )
    model_stage: str = Field(default="Staging", description="MLflow model stage")
    best_params_path: str = Field(..., description="Path to sweep best_params.yaml")
    target_column: str = Field(default="median_house_value", description="Target column name")

    wandb_project: str = Field(..., description="W&B project name")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)
