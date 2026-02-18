"""Pydantic models for model selection."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ModelSelectionConfig(BaseModel):
    """Configuration for model selection."""

    train_artifact_name: str
    test_artifact_name: str
    gcs_train_path: str
    gcs_test_path: str
    bucket_name: str
    wandb_project: str
    target_column: str = "median_house_value"
    random_state: int = 42
    artifact_name: str = "model_selection_results"
    artifact_type: str = "model_results"
    artifact_description: str = ""

    @field_validator("random_state")
    @classmethod
    def validate_random_state(cls, value: int) -> int:
        """Validate random state."""
        if value < 0:
            raise ValueError("random_state must be non-negative")
        return value


class ModelCvMetrics(BaseModel):
    """Cross-validation metrics for a trained model."""

    mean_test_score: float = Field(..., description="Mean validation MAE")
    std_test_score: float = Field(..., description="Validation MAE std deviation")
    mean_train_score: float = Field(..., description="Mean train MAE")
    std_train_score: float = Field(..., description="Train MAE std deviation")


class ModelMetrics(BaseModel):
    """Model performance metrics with business focus."""

    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    r2: float = Field(..., description="R2 Score")
    mape: float = Field(..., description="Mean Absolute Percentage Error (%)")
    smape: float = Field(..., description="Symmetric MAPE (%)")
    wmape: float = Field(..., description="Weighted MAPE (%)")
    median_ape: float = Field(..., description="Median Absolute Percentage Error (%)")
    within_5pct: float = Field(..., description="% predictions within 5% of actual")
    within_10pct: float = Field(..., description="% predictions within 10% of actual")
    within_15pct: float = Field(..., description="% predictions within 15% of actual")


class ModelResult(BaseModel):
    """Result for a single model."""

    model_name: str
    metrics: ModelMetrics
    cv_metrics: ModelCvMetrics
    best_params: dict[str, Any]
    training_time: float


class ModelSelectionResult(BaseModel):
    """Overall model selection result."""

    best_model_name: str
    all_results: dict[str, dict[str, Any]]
    total_models_trained: int
    train_samples: int
    test_samples: int
    num_features: int
    target_column: str


class BestModelSummary(BaseModel):
    """Serialized metadata used by downstream sweep step."""

    best_model_name: str
    metrics: ModelMetrics
    cv_metrics: ModelCvMetrics
    best_params: dict[str, Any]
    training_time: float
    source_step: str = "05_model_selection"


class ModelSelectionArtifact(BaseModel):
    """Summary file persisted by model selection step."""

    best_model: BestModelSummary
    all_model_names: list[str]
