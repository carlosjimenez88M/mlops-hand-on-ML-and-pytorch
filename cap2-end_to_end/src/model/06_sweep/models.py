"""
Pydantic models for sweep configuration and results.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class SweepConfig(BaseModel):
    """W&B Sweep configuration."""

    train_artifact_name: str = Field(..., description="Training data artifact name")
    test_artifact_name: str = Field(..., description="Test data artifact name")
    gcs_train_path: str = Field(..., description="GCS path to training data")
    gcs_test_path: str = Field(..., description="GCS path to test data")
    bucket_name: str = Field(..., description="GCS bucket name")
    wandb_project: str = Field(..., description="W&B project name")
    best_model_type: str = Field(..., description="Type of model from selection step")
    target_column: str = Field(default="median_house_value", description="Target column")
    random_state: int = Field(default=42, description="Random state")
    sweep_count: int = Field(default=50, description="Number of sweep runs")


class SweepResult(BaseModel):
    """Results from W&B Sweep."""

    sweep_id: str = Field(..., description="W&B sweep ID")
    best_run_id: str = Field(..., description="Best run ID")
    best_params: Dict[str, Any] = Field(..., description="Best hyperparameters")
    best_mape: float = Field(..., description="Best MAPE score")
    best_within_10pct: Optional[float] = Field(None, description="% predictions within 10%")
    sweep_url: str = Field(..., description="W&B sweep URL")
