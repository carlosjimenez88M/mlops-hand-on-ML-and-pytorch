# components/01_download_data/models.py
"""
Pydantic models para validación y type safety
Author: Carlos Daniel Hernandez
Date: 2025-11-25
"""

#=======================#
# ----- Libraries ----- #
#=======================#

from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, HttpUrl, field_validator, ConfigDict

#=========================#
# ---- Class Methods ---- #
#=========================#

class DownloadConfig(BaseModel):
    """Configuration for data download."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    file_url: HttpUrl = Field(
        ...,
        description="URL to download the data from"
    )
    
    artifact_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the artifact in W&B"
    )
    
    artifact_type: str = Field(
        ...,
        min_length=1,
        description="Type of artifact (raw_data, processed_data, etc.)"
    )
    
    artifact_description: str = Field(
        default="",
        description="Description of the artifact"
    )
    
    gcs_output_path: str = Field(
        default="data/01-raw/housing.csv",
        description="Path in GCS (without gs://bucket/)"
    )
    
    bucket_name: str = Field(
        ...,
        min_length=3,
        max_length=63,
        description="Name of the GCS bucket (without gs://)"
    )
    
    wandb_project: str = Field(
        default="housing-mlops-gcp",
        description="Name of the project in W&B"
    )
    
    @field_validator('bucket_name')
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validates that the bucket name is valid for GCS."""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError(
                "Bucket name must contain " \
                "only letters, numbers, hyphens, and underscores"  
            )
        if v.startswith('-') or v.endswith('-'):
            raise ValueError("Bucket name must not start or end with a hyphen")
        return v.lower()
    
    @field_validator('gcs_output_path')
    @classmethod
    def validate_gcs_path(cls, v: str) -> str:
        """Ensures the path does not start with / or gs://."""
        v = v.lstrip('/')
        if v.startswith('gs://'):
            raise ValueError("gcs_output_path must not include gs:// or the bucket name")
        if not any(v.endswith(ext) for ext in ['.csv', '.parquet', '.json']):
            raise ValueError(f"Unsupported extension in gcs_output_path: {v}")
        return v


class FileStats(BaseModel):
    """File statistics of the downloaded file."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    file_size_mb: float = Field(..., ge=0, description="Size in MB")
    n_rows: Optional[int] = Field(None, ge=0, description="Number of rows")
    n_columns: Optional[int] = Field(None, ge=0, description="Number of columns")
    columns: Optional[List[str]] = Field(None, description="List of columns")
    missing_values: Optional[Dict[str, int]] = Field(
        None,
        description="Missing values per column"
    )
    downloaded_at: datetime = Field(
        default_factory=datetime.now,
        description="Download timestamp"
    )
    
    @property
    def has_missing_values(self) -> bool:
        """Checks if there are missing values."""
        if not self.missing_values:
            return False
        return sum(self.missing_values.values()) > 0
    
    def missing_percentage(self, column: str) -> Optional[float]:
        """Calculates the percentage of missing values for a column."""
        if not self.missing_values or not self.n_rows or column not in self.missing_values:
            return None
        return (self.missing_values[column] / self.n_rows) * 100


class DownloadResult(BaseModel):
    """Download result model."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    gcs_uri: str = Field(..., description="Full GCS URI")
    stats: FileStats = Field(..., description="File statistics")
    artifact_name: str = Field(..., description="W&B artifact name")
    success: bool = Field(default=True, description="Whether the operation was successful")
    error_message: Optional[str] = Field(None, description="Error message if applicable")
    
    @property
    def gcs_path(self) -> str:
        """Extracts the relative path from the GCS URI."""
        return self.gcs_uri.replace(f"gs://", "").split("/", 1)[1]
    
    @property
    def bucket_name(self) -> str:
        """Extracts the bucket name from the URI."""
        return self.gcs_uri.replace("gs://", "").split("/")[0]


class WandBArtifactMetadata(BaseModel):
    """Metadata for W&B artifact."""
    
    original_url: str
    gcs_uri: str
    bucket: str
    file_size_mb: float
    n_rows: Optional[int] = None
    n_columns: Optional[int] = None
    downloaded_at: str
    component: str = "01_download_data"
    storage_type: str = "gcs_only"
    
    def to_dict(self) -> dict:
        """Converts to dictionary removing None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}