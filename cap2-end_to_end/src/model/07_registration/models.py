"""Pydantic models for model registration."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SupportedModelType(str, Enum):
    """Supported model types across selection/sweep/registration steps."""

    RANDOM_FOREST = "RandomForest"
    GRADIENT_BOOSTING = "GradientBoosting"
    RIDGE = "Ridge"
    LASSO = "Lasso"
    DECISION_TREE = "DecisionTree"


class RegistrationRuntimeConfig(BaseModel):
    """Validated runtime config parsed from CLI arguments."""

    bucket_name: str = Field(..., description="GCS bucket name")
    gcs_train_path: str = Field(..., description="Path to training data in GCS")
    gcs_test_path: str = Field(..., description="Path to test data in GCS")
    best_params_path: str = Field(..., description="Path to best_params.yaml")
    registered_model_name: str = Field(default="housing_price_model")
    model_stage: str = Field(default="Staging")
    target_column: str = Field(default="median_house_value")
    wandb_project: str = Field(..., description="W&B project name")

    @field_validator("bucket_name", "gcs_train_path", "gcs_test_path", "wandb_project")
    @classmethod
    def validate_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be empty")
        return stripped


class SweepBestParamsFile(BaseModel):
    """Content schema for best_params.yaml generated in step 06."""

    sweep_id: str
    best_run_id: str
    best_run_name: str
    model_type: SupportedModelType = SupportedModelType.RANDOM_FOREST
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    sweep_url: str


class RegistrationResult(BaseModel):
    """Result from model registration."""

    model_name: str = Field(..., description="Registered model name")
    model_version: str = Field(..., description="Model version number")
    model_stage: str = Field(..., description="Model stage (Staging/Production)")
    model_uri: str = Field(..., description="MLflow model URI")
    gcs_model_uri: str = Field(..., description="GCS model URI")
    run_id: str = Field(..., description="MLflow run ID")
    model_type: SupportedModelType = Field(..., description="Model family")
    hyperparameters: dict[str, Any] = Field(..., description="Model hyperparameters")
    metrics: dict[str, float] = Field(..., description="Model metrics")
    feature_columns: list[str] = Field(..., description="List of feature columns")
