"""API Configuration"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Housing Price Prediction API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Model - MLflow (priority)
    MLFLOW_MODEL_NAME: str = ""
    MLFLOW_MODEL_STAGE: str = "Production"
    MLFLOW_TRACKING_URI: str = ""

    # Model - GCS (fallback)
    GCS_BUCKET: str = ""
    GCS_MODEL_PATH: str = "models/trained/housing_price_model.pkl"

    # Model - Local (fallback)
    LOCAL_MODEL_PATH: str = "models/trained/housing_price_model.pkl"

    # Weights & Biases
    WANDB_API_KEY: str = ""
    WANDB_PROJECT: str = "housing-mlops-api"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
