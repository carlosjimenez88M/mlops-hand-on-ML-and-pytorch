"""
Configuration for hyperparameter sweep.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class SweepConfig(BaseSettings):
    """Configuration for W&B Sweep."""

    # W&B Configuration
    wandb_project: str = Field(default="housing-mlops-gcp", description="W&B project name")
    wandb_entity: str = Field(default="", description="W&B entity/team name")

    # GCS Configuration
    bucket_name: str = Field(..., description="GCS bucket name")
    train_data_path: str = Field(..., description="Path to training data in GCS")
    test_data_path: str = Field(..., description="Path to test data in GCS")

    # Model Configuration
    model_artifact_name: str = Field(..., description="W&B artifact name for best model")
    best_model_type: str = Field(..., description="Type of best model (e.g., randomforest)")

    # Sweep Configuration
    sweep_count: int = Field(default=50, description="Number of sweep runs")
    metric_name: str = Field(default="mape", description="Metric to optimize")
    metric_goal: str = Field(default="minimize", description="Optimization goal (minimize/maximize)")

    # Target
    target_column: str = Field(default="median_house_value", description="Target column name")
    random_state: int = Field(default=42, description="Random state for reproducibility")

    class Config:
        env_file = ".env"
        case_sensitive = True
