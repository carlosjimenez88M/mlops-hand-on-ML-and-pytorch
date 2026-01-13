"""
Pydantic models for feature engineering module
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering component."""

    input_artifact_name: str = Field(..., description="Name of the input artifact from W&B")
    gcs_input_path: str = Field(..., description="GCS path to input data")
    gcs_output_path: str = Field(..., description="GCS path for processed data")
    artifact_name: str = Field(..., description="Name of the output artifact in W&B")
    artifact_type: str = Field(default="engineered_features", description="Type of artifact")
    artifact_description: str = Field(default="", description="Artifact description")
    bucket_name: str = Field(..., description="GCS bucket name")
    wandb_project: str = Field(..., description="W&B Project")
    n_clusters: int = Field(default=10, description="Number of clusters for geo features")
    gamma: float = Field(default=0.1, description="Gamma parameter for clustering")
    random_state: int = Field(default=42, description="Random state for reproducibility")

    @field_validator("n_clusters")
    def validate_n_clusters(cls, v):
        if v < 2 or v > 50:
            raise ValueError("n_clusters must be between 2 and 50")
        return v

    @field_validator("gamma")
    def validate_gamma(cls, v):
        if v <= 0:
            raise ValueError("gamma must be positive")
        return v


class TransformationResult(BaseModel):
    """Result of feature engineering transformation."""

    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    numerical_features: List[str]
    categorical_features: List[str]
    geo_features: List[str]
    target_column: str
    features_added: int
    rows_processed: int
