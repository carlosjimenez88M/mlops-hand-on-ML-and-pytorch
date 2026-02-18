"""
Pydantic models for data segregation module
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

from pydantic import BaseModel, Field, field_validator


class SegregationConfig(BaseModel):
    """Configuration for data segregation component."""

    input_artifact_name: str = Field(..., description="Name of the input artifact from W&B")
    gcs_input_path: str = Field(..., description="GCS path to input data")
    gcs_train_output_path: str = Field(..., description="GCS path for train data")
    gcs_test_output_path: str = Field(..., description="GCS path for test data")
    artifact_root: str = Field(..., description="Root name for output artifacts")
    artifact_type: str = Field(default="segregated_data", description="Type of artifact")
    artifact_description: str = Field(default="", description="Artifact description")
    bucket_name: str = Field(..., description="GCS bucket name")
    wandb_project: str = Field(..., description="W&B Project")
    test_size: float = Field(default=0.2, description="Fraction of data for test set")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    target_column: str = Field(default="median_house_value", description="Target column name")

    @field_validator("test_size")
    def validate_test_size(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("test_size must be between 0 and 1")
        return v

    @field_validator("random_state")
    def validate_random_state(cls, v):
        if v < 0:
            raise ValueError("random_state must be non-negative")
        return v


class SegregationResult(BaseModel):
    """Result of data segregation."""

    input_shape: tuple[int, int]
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    target_column: str
    test_size: float
    train_gcs_uri: str
    test_gcs_uri: str
    train_samples: int
    test_samples: int
