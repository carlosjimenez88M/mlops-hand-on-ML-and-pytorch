"""Utility functions for final model training and registration."""

from __future__ import annotations

import io
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from models import SupportedModelType

logger = logging.getLogger(__name__)

EPSILON = 1e-8


class GCSDataRepository:
    """Simple data repository for GCS-hosted train/test datasets."""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") == "":
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)

    def download_dataframe(self, gcs_path: str) -> pd.DataFrame:
        """Download CSV/Parquet from GCS."""
        logger.info("Downloading from GCS: gs://%s/%s", self.bucket_name, gcs_path)

        blob = self._bucket.blob(gcs_path)
        content = blob.download_as_bytes()

        if gcs_path.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))

        logger.info("Loaded DataFrame: %s rows, %s columns", df.shape[0], df.shape[1])
        return df

    def upload_pickle(self, obj: Any, gcs_path: str) -> str:
        """Serialize object as pickle and upload it to GCS."""
        logger.info("Uploading model to GCS: gs://%s/%s", self.bucket_name, gcs_path)

        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        buffer.seek(0)

        blob = self._bucket.blob(gcs_path)
        blob.upload_from_file(buffer, content_type="application/octet-stream")

        uri = f"gs://{self.bucket_name}/{gcs_path}"
        logger.info("Model uploaded to %s", uri)
        return uri


class ModelFactory:
    """Create final estimator from tuned hyperparameters."""

    @staticmethod
    def build_model(model_type: SupportedModelType, params: dict[str, Any]) -> RegressorMixin:
        """Instantiate model object by model type."""
        if model_type == SupportedModelType.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=(None if params.get("max_depth") is None else int(params["max_depth"])),
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=params.get("max_features"),
                random_state=int(params.get("random_state", 42)),
                n_jobs=-1,
            )

        if model_type == SupportedModelType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(
                n_estimators=int(params["n_estimators"]),
                learning_rate=float(params["learning_rate"]),
                max_depth=int(params["max_depth"]),
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                subsample=float(params["subsample"]),
                max_features=params.get("max_features"),
                random_state=int(params.get("random_state", 42)),
            )

        if model_type == SupportedModelType.RIDGE:
            return Ridge(
                alpha=float(params["alpha"]), random_state=int(params.get("random_state", 42))
            )

        if model_type == SupportedModelType.LASSO:
            return Lasso(
                alpha=float(params["alpha"]),
                random_state=int(params.get("random_state", 42)),
                max_iter=20000,
            )

        if model_type == SupportedModelType.DECISION_TREE:
            return DecisionTreeRegressor(
                max_depth=(None if params.get("max_depth") is None else int(params["max_depth"])),
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=params.get("max_features"),
                random_state=int(params.get("random_state", 42)),
            )

        raise ValueError(f"Unsupported model type: {model_type}")


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE with numerical stability."""
    denominator = np.where(np.abs(y_true) < EPSILON, EPSILON, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric MAPE."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator < EPSILON, EPSILON, denominator)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def weighted_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Weighted MAPE."""
    denominator = max(float(np.sum(np.abs(y_true))), EPSILON)
    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100)


def predictions_within_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.10,
) -> float:
    """Calculate prediction accuracy inside relative error threshold."""
    denominator = np.where(np.abs(y_true) < EPSILON, EPSILON, np.abs(y_true))
    errors = np.abs(y_pred - y_true) / denominator
    return float((errors < threshold).mean() * 100)


def prepare_data(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    x = df.drop(columns=[target_column])
    y = df[target_column]

    logger.info("Prepared data: %s samples, %s features", x.shape[0], x.shape[1])
    return x, y


def train_final_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
    model_type: SupportedModelType,
) -> RegressorMixin:
    """Train final model with optimized hyperparameters."""
    logger.info("=" * 70)
    logger.info("TRAINING FINAL MODEL")
    logger.info("=" * 70)
    logger.info("Model type: %s", model_type.value)
    logger.info("Hyperparameters: %s", params)

    model = ModelFactory.build_model(model_type=model_type, params=params)
    model.fit(x_train, y_train)

    logger.info("Training completed")
    return model


def evaluate_model(
    model: RegressorMixin,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate model and return comprehensive metrics."""
    logger.info("Evaluating model...")

    y_pred = np.asarray(model.predict(x_test), dtype=float)
    y_true = y_test.to_numpy(dtype=float)

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        "wmape": weighted_mean_absolute_percentage_error(y_true, y_pred),
        "median_ape": float(
            np.median(
                np.abs(y_true - y_pred)
                / np.where(np.abs(y_true) < EPSILON, EPSILON, np.abs(y_true))
            )
            * 100
        ),
        "within_5pct": predictions_within_threshold(y_true, y_pred, 0.05),
        "within_10pct": predictions_within_threshold(y_true, y_pred, 0.10),
        "within_15pct": predictions_within_threshold(y_true, y_pred, 0.15),
    }

    logger.info("Final model metrics: %s", metrics)
    return metrics


def save_model_locally(model: RegressorMixin, output_path: Path) -> None:
    """Save trained model locally."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        pickle.dump(model, file)

    logger.info("Model saved to: %s", output_path)


def _extract_importance_values(model: RegressorMixin) -> np.ndarray | None:
    """Extract feature importance/coefficients depending on model family."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(getattr(model, "feature_importances_"), dtype=float)

    if hasattr(model, "coef_"):
        coefficients = np.asarray(getattr(model, "coef_"), dtype=float)
        return np.abs(coefficients).reshape(-1)

    return None


def create_feature_importance_plot(
    model: RegressorMixin,
    feature_names: list[str],
    output_dir: Path = Path("artifacts/registration"),
) -> Path | None:
    """Create feature importance plot when model exposes importances/coefficients."""
    try:
        importance_values = _extract_importance_values(model)
        if importance_values is None or len(importance_values) != len(feature_names):
            logger.info("Model does not expose feature importances; skipping plot")
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "feature_importance.png"

        order = np.argsort(importance_values)[::-1]
        top_n = min(15, len(feature_names))
        top_indices = order[:top_n]

        top_features = [feature_names[index] for index in top_indices]
        top_values = importance_values[top_indices]

        fig, axis = plt.subplots(figsize=(10, 8))
        y_positions = np.arange(len(top_features))

        axis.barh(y_positions, top_values, align="center", color="steelblue")
        axis.set_yticks(y_positions)
        axis.set_yticklabels(top_features)
        axis.invert_yaxis()
        axis.set_xlabel("Feature Importance")
        axis.set_title("Top 15 Most Important Features")
        axis.grid(axis="x", alpha=0.3)

        for index, value in enumerate(top_values):
            axis.text(value + 0.001, index, f"{value:.4f}", va="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Feature importance plot saved to: %s", plot_path)
        return plot_path

    except Exception as error:
        logger.warning("Failed to create feature importance plot: %s", error)
        return None
