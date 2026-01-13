"""
Pydantic models for preprocessing and imputation module
Author: Carlos Daniel Jiménez
Date: 2025-11-28
"""

from datetime import datetime
from typing import Optional, Dict, List, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing."""

    model_config = ConfigDict(validate_assignment=True)

    # Input data
    input_artifact_name: str = Field(
        ...,
        min_length=1,
        description="Name of the input artifact from W&B"
    )

    gcs_input_path: str = Field(
        ...,
        min_length=1,
        description="GCS path to input data (without gs://bucket/)"
    )

    # Output data
    gcs_output_path: str = Field(
        default="data/02-processed/housing_processed.csv",
        description="Path in GCS for processed data"
    )

    artifact_name: str = Field(
        default="housing_data_processed",
        description="Name of the output artifact in W&B"
    )

    artifact_type: str = Field(
        default="processed_data",
        description="Type of artifact"
    )

    artifact_description: str = Field(
        default="",
        description="Description of the artifact"
    )

    # Bucket info
    bucket_name: str = Field(
        ...,
        min_length=3,
        max_length=63,
        description="Name of the GCS bucket"
    )

    # W&B config
    wandb_project: str = Field(
        default="housing-mlops-gcp",
        description="Name of the project in W&B"
    )

    # Preprocessing options
    imputation_strategy: Literal["mean", "median", "mode", "drop", "auto"] = Field(
        default="auto",
        description="Strategy for handling missing values. 'auto' compares all methods and selects the best one."
    )

    create_features: bool = Field(
        default=True,
        description="Whether to create engineered features"
    )

    @field_validator('bucket_name')
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validates that the bucket name is valid for GCS."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError(
                "Bucket name must contain only letters, numbers, hyphens, and underscores"
            )
        if v.startswith('-') or v.endswith('-'):
            raise ValueError("Bucket name must not start or end with a hyphen")
        return v.lower()


class PreprocessingStats(BaseModel):
    """Statistics about the preprocessing operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input stats
    input_rows: int = Field(..., ge=0)
    input_columns: int = Field(..., ge=0)
    missing_values_before: Dict[str, int] = Field(default_factory=dict)

    # Output stats
    output_rows: int = Field(..., ge=0)
    output_columns: int = Field(..., ge=0)
    missing_values_after: Dict[str, int] = Field(default_factory=dict)

    # Feature engineering
    new_features: List[str] = Field(default_factory=list)

    # File info
    input_size_mb: float = Field(..., ge=0)
    output_size_mb: float = Field(..., ge=0)

    processed_at: datetime = Field(
        default_factory=datetime.now,
        description="Processing timestamp"
    )

    @property
    def rows_dropped(self) -> int:
        """Calculate rows dropped."""
        return self.input_rows - self.output_rows

    @property
    def columns_added(self) -> int:
        """Calculate columns added."""
        return self.output_columns - self.input_columns


class PreprocessingResult(BaseModel):
    """Result of the preprocessing operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gcs_input_uri: str = Field(..., description="Input GCS URI")
    gcs_output_uri: str = Field(..., description="Output GCS URI")
    stats: PreprocessingStats = Field(..., description="Preprocessing statistics")
    artifact_name: str = Field(..., description="W&B artifact name")
    success: bool = Field(default=True, description="Whether the operation was successful")
    error_message: Optional[str] = Field(None, description="Error message if applicable")


class WandBArtifactMetadata(BaseModel):
    """Metadata for W&B artifact."""

    input_artifact: str
    input_gcs_uri: str
    output_gcs_uri: str
    bucket: str
    input_size_mb: float
    output_size_mb: float
    input_rows: int
    output_rows: int
    rows_dropped: int
    input_columns: int
    output_columns: int
    columns_added: int
    new_features: List[str]
    imputation_strategy: str
    processed_at: str
    component: str = "02_preprocessing_and_imputation"

    def to_dict(self) -> dict:
        """Converts to dictionary removing None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
