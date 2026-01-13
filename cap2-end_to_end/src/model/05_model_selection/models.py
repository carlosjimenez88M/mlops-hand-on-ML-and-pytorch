"""
Pydantic models for model selection
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Tuple


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
    def validate_random_state(cls, v):
        if v < 0:
            raise ValueError("random_state must be non-negative")
        return v


class ModelMetrics(BaseModel):
    """Model performance metrics with business focus."""

    # Traditional metrics
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    r2: float = Field(..., description="R² Score")

    # Business-focused metrics
    mape: float = Field(..., description="Mean Absolute Percentage Error (%)")
    median_ape: float = Field(..., description="Median Absolute Percentage Error (%)")
    within_5pct: float = Field(..., description="% predictions within 5% of actual")
    within_10pct: float = Field(..., description="% predictions within 10% of actual")
    within_15pct: float = Field(..., description="% predictions within 15% of actual")


class ModelResult(BaseModel):
    """Result for a single model."""

    model_name: str
    metrics: ModelMetrics
    best_params: Dict[str, Any]
    training_time: float


class ModelSelectionResult(BaseModel):
    """Overall model selection result."""

    best_model_name: str
    all_results: Dict[str, Dict[str, Any]]
    total_models_trained: int
    train_samples: int
    test_samples: int
    num_features: int
    target_column: str
