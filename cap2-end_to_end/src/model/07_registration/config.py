"""
Configuration for model registration.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class RegistrationConfig(BaseSettings):
    """Configuration for model registration."""

    # Data paths
    bucket_name: str = Field(..., description="GCS bucket name")
    gcs_train_path: str = Field(..., description="Path to training data in GCS")
    gcs_test_path: str = Field(..., description="Path to test data in GCS")

    # Model configuration
    registered_model_name: str = Field(default="housing_price_model", description="Name for registered model")
    model_stage: str = Field(default="Staging", description="MLflow model stage")

    # Best params path
    best_params_path: str = Field(..., description="Path to best_params.yaml from sweep")

    # Target
    target_column: str = Field(default="median_house_value", description="Target column name")

    # W&B
    wandb_project: str = Field(..., description="W&B project name")

    class Config:
        env_file = ".env"
        case_sensitive = True
