"""
Pydantic models for model registration.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, List


class RegistrationResult(BaseModel):
    """Result from model registration."""

    model_name: str = Field(..., description="Registered model name")
    model_version: str = Field(..., description="Model version number")
    model_stage: str = Field(..., description="Model stage (Staging/Production)")
    model_uri: str = Field(..., description="MLflow model URI")
    run_id: str = Field(..., description="MLflow run ID")
    hyperparameters: Dict[str, Any] = Field(..., description="Model hyperparameters")
    metrics: Dict[str, float] = Field(..., description="Model metrics")
    feature_columns: List[str] = Field(..., description="List of feature columns")
