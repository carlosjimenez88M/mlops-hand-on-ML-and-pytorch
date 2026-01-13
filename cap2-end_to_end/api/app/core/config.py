"""API Configuration"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Housing Price Prediction API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Model
    MODEL_PATH: str = "models/best_model.pkl"
    GCS_BUCKET: str = ""
    GCS_MODEL_PATH: str = "models/05-selection/randomforest_best.pkl"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
