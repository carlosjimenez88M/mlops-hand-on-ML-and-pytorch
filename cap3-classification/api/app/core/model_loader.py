"""Model loading and prediction service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from cap3_classification.utils.io import read_json
from cap3_classification.utils.logging import get_logger

logger = get_logger(__name__)


class ModelService:
    """Load the registered model and serve predictions."""

    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        model_alias: str,
        local_model_path: str,
        serving_metadata_path: str,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.model_alias = model_alias
        self.local_model_path = Path(local_model_path)
        self.serving_metadata_path = Path(serving_metadata_path)
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}

    def load(self) -> None:
        """Load metadata and the model from MLflow or local fallback."""
        if self.serving_metadata_path.exists():
            self.metadata = read_json(self.serving_metadata_path)

        mlflow.set_tracking_uri(self.tracking_uri)
        model_uri = f"models:/{self.model_name}@{self.model_alias}"
        try:
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info("Loaded model from MLflow URI %s", model_uri)
            return
        except Exception as error:
            logger.warning("Falling back to local model because MLflow load failed: %s", error)

        if not self.local_model_path.exists():
            raise FileNotFoundError(f"Local fallback model not found: {self.local_model_path}")
        self.model = joblib.load(self.local_model_path)
        logger.info("Loaded model from local fallback %s", self.local_model_path)

    @property
    def expected_input_dim(self) -> int | None:
        value = self.metadata.get("input_dim")
        return int(value) if value is not None else None

    def predict(self, instances: list[list[float]]) -> dict[str, Any]:
        """Predict labels and probabilities for one or more flattened images."""
        if self.model is None:
            raise RuntimeError("Model has not been loaded")
        features = np.asarray(instances, dtype=float)
        if self.expected_input_dim is not None and features.shape[1] != self.expected_input_dim:
            raise ValueError(
                f"Expected {self.expected_input_dim} features but received {features.shape[1]}"
            )
        predictions = self.model.predict(features).tolist()
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features).tolist()
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "model_name": self.metadata.get("registered_model_name", self.model_name),
            "model_version": self.metadata.get("model_version", "local"),
        }
