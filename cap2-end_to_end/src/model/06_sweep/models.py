"""Pydantic models for sweep execution and outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SupportedModelType(str, Enum):
    """Supported estimators for Bayesian sweep."""

    RANDOM_FOREST = "RandomForest"
    GRADIENT_BOOSTING = "GradientBoosting"
    RIDGE = "Ridge"
    LASSO = "Lasso"
    DECISION_TREE = "DecisionTree"


class SweepExecutionConfig(BaseModel):
    """Validated CLI inputs for sweep step."""

    train_artifact_name: str = Field(..., description="Training data artifact name")
    test_artifact_name: str = Field(..., description="Test data artifact name")
    gcs_train_path: str = Field(..., description="GCS path to training data")
    gcs_test_path: str = Field(..., description="GCS path to test data")
    bucket_name: str = Field(..., description="GCS bucket name")
    wandb_project: str = Field(..., description="W&B project name")
    best_model_type: SupportedModelType = Field(
        default=SupportedModelType.RANDOM_FOREST,
        description="Base model selected in step 05",
    )
    target_column: str = Field(default="median_house_value", description="Target column")
    sweep_count: int = Field(default=50, ge=1, description="Number of sweep runs")
    sweep_config: str = Field(
        default="sweep_config.yaml", description="Optional custom sweep config"
    )
    random_state: int = Field(default=42, ge=0, description="Random state for reproducibility")

    @field_validator("gcs_train_path", "gcs_test_path", "bucket_name", "wandb_project")
    @classmethod
    def validate_not_empty(cls, value: str) -> str:
        """Ensure required string fields are not empty."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be empty")
        return stripped


class SweepRunSummary(BaseModel):
    """Single run summary captured during sweep agent execution."""

    run_id: str
    run_name: str
    entity: str | None = None
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]


class SweepBestParams(BaseModel):
    """Persisted output used by registration step."""

    sweep_id: str
    best_run_id: str
    best_run_name: str
    model_type: SupportedModelType
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    sweep_url: str
