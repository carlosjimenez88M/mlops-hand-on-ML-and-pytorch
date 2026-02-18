"""Environment-backed configuration for sweep step (optional helper)."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SweepConfig(BaseSettings):
    """Configuration for W&B sweep execution."""

    wandb_project: str = Field(default="housing-mlops-gcp", description="W&B project name")
    wandb_entity: str = Field(default="", description="W&B entity/team name")

    bucket_name: str = Field(..., description="GCS bucket name")
    train_data_path: str = Field(..., description="Path to training data in GCS")
    test_data_path: str = Field(..., description="Path to test data in GCS")

    best_model_type: str = Field(default="RandomForest", description="Selected base model")
    sweep_count: int = Field(default=50, description="Number of sweep runs")
    target_column: str = Field(default="median_house_value", description="Target column")
    random_state: int = Field(default=42, description="Random state")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)
