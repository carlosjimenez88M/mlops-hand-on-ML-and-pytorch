"""Utility classes and functions for Bayesian hyperparameter sweep."""

from __future__ import annotations

import io
import logging
import os
from typing import Any

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
    """Thin repository class for reading tabular data from GCS."""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") == "":
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)

    def download_dataframe(self, gcs_path: str) -> pd.DataFrame:
        """Download CSV/Parquet file from GCS into DataFrame."""
        logger.info("Downloading from GCS: gs://%s/%s", self.bucket_name, gcs_path)

        blob = self._bucket.blob(gcs_path)
        content = blob.download_as_bytes()

        if gcs_path.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))

        logger.info("Loaded DataFrame: %s rows, %s columns", df.shape[0], df.shape[1])
        return df


class ModelFactory:
    """Create estimators, sanitize hyperparameters and generate sweep spaces."""

    _allowed_keys: dict[SupportedModelType, set[str]] = {
        SupportedModelType.RANDOM_FOREST: {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        },
        SupportedModelType.GRADIENT_BOOSTING: {
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "subsample",
            "max_features",
        },
        SupportedModelType.RIDGE: {"alpha"},
        SupportedModelType.LASSO: {"alpha"},
        SupportedModelType.DECISION_TREE: {
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        },
    }

    _int_keys: dict[SupportedModelType, set[str]] = {
        SupportedModelType.RANDOM_FOREST: {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
        },
        SupportedModelType.GRADIENT_BOOSTING: {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
        },
        SupportedModelType.RIDGE: set(),
        SupportedModelType.LASSO: set(),
        SupportedModelType.DECISION_TREE: {
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
        },
    }

    @classmethod
    def sanitize_params(
        cls,
        model_type: SupportedModelType,
        raw_params: dict[str, Any],
        random_state: int,
    ) -> dict[str, Any]:
        """Convert sweep config values into sklearn-friendly params."""
        allowed_keys = cls._allowed_keys[model_type]
        cleaned: dict[str, Any] = {}

        for key in allowed_keys:
            if key not in raw_params:
                continue

            value = raw_params[key]

            if key in cls._int_keys[model_type] and value is not None:
                value = int(value)

            if key == "max_depth":
                if value in (0, "0"):
                    value = None
                if isinstance(value, str) and value.lower() in {"none", "null"}:
                    value = None

            if key == "max_features" and isinstance(value, str):
                if value.lower() in {"none", "null"}:
                    value = None

            cleaned[key] = value

        cleaned["random_state"] = random_state
        return cleaned

    @classmethod
    def required_param_keys(cls, model_type: SupportedModelType) -> set[str]:
        """Return required params for a model type."""
        return set(cls._allowed_keys[model_type])

    @staticmethod
    def build_model(
        model_type: SupportedModelType,
        params: dict[str, Any],
    ) -> RegressorMixin:
        """Instantiate model for the selected estimator type."""
        if model_type == SupportedModelType.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params.get("max_depth"),
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params.get("max_features"),
                random_state=params.get("random_state", 42),
                n_jobs=-1,
            )

        if model_type == SupportedModelType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                subsample=params["subsample"],
                max_features=params.get("max_features"),
                random_state=params.get("random_state", 42),
            )

        if model_type == SupportedModelType.RIDGE:
            return Ridge(alpha=params["alpha"], random_state=params.get("random_state", 42))

        if model_type == SupportedModelType.LASSO:
            return Lasso(
                alpha=params["alpha"],
                random_state=params.get("random_state", 42),
                max_iter=20000,
            )

        if model_type == SupportedModelType.DECISION_TREE:
            return DecisionTreeRegressor(
                max_depth=params.get("max_depth"),
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params.get("max_features"),
                random_state=params.get("random_state", 42),
            )

        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def build_default_sweep_config(model_type: SupportedModelType) -> dict[str, Any]:
        """Return Bayesian sweep configuration for the selected estimator."""
        metric_name = "wmape"
        goal = "minimize"

        base = {
            "method": "bayes",
            "metric": {"name": metric_name, "goal": goal},
            "early_terminate": {"type": "hyperband", "min_iter": 8, "eta": 3, "s": 2},
        }

        if model_type == SupportedModelType.RANDOM_FOREST:
            base["parameters"] = {
                "n_estimators": {"min": 50, "max": 500},
                "max_depth": {"min": 5, "max": 35},
                "min_samples_split": {"min": 2, "max": 20},
                "min_samples_leaf": {"min": 1, "max": 10},
                "max_features": {"values": ["sqrt", "log2"]},
            }
            base["name"] = "housing-rf-bayes"
            return base

        if model_type == SupportedModelType.GRADIENT_BOOSTING:
            base["parameters"] = {
                "n_estimators": {"min": 50, "max": 450},
                "learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 0.005,
                    "max": 0.3,
                },
                "max_depth": {"min": 2, "max": 8},
                "min_samples_split": {"min": 2, "max": 20},
                "min_samples_leaf": {"min": 1, "max": 10},
                "subsample": {"min": 0.6, "max": 1.0},
                "max_features": {"values": ["sqrt", "log2", None]},
            }
            base["name"] = "housing-gb-bayes"
            return base

        if model_type == SupportedModelType.RIDGE:
            base["parameters"] = {
                "alpha": {
                    "distribution": "log_uniform_values",
                    "min": 0.0001,
                    "max": 100.0,
                }
            }
            base["name"] = "housing-ridge-bayes"
            return base

        if model_type == SupportedModelType.LASSO:
            base["parameters"] = {
                "alpha": {
                    "distribution": "log_uniform_values",
                    "min": 0.0001,
                    "max": 100.0,
                }
            }
            base["name"] = "housing-lasso-bayes"
            return base

        base["parameters"] = {
            "max_depth": {"min": 3, "max": 40},
            "min_samples_split": {"min": 2, "max": 20},
            "min_samples_leaf": {"min": 1, "max": 10},
            "max_features": {"values": ["sqrt", "log2", None]},
        }
        base["name"] = "housing-dt-bayes"
        return base


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE with numerical stability."""
    denominator = np.where(np.abs(y_true) < EPSILON, EPSILON, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate SMAPE with numerical stability."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator < EPSILON, EPSILON, denominator)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def weighted_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate weighted MAPE with numerical stability."""
    denominator = max(float(np.sum(np.abs(y_true))), EPSILON)
    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100)


def predictions_within_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.10,
) -> float:
    """Calculate percentage of predictions inside relative-error threshold."""
    denominator = np.where(np.abs(y_true) < EPSILON, EPSILON, np.abs(y_true))
    errors = np.abs(y_pred - y_true) / denominator
    return float((errors < threshold).mean() * 100)


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: SupportedModelType,
    params: dict[str, Any],
) -> RegressorMixin:
    """Train selected model type with provided hyperparameters."""
    logger.info("Training %s with params: %s", model_type.value, params)
    model = ModelFactory.build_model(model_type=model_type, params=params)
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: RegressorMixin,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate model with business and regression metrics."""
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

    logger.info(
        "MAPE=%.2f%% | SMAPE=%.2f%% | wMAPE=%.2f%% | Within10%%=%.1f%%",
        metrics["mape"],
        metrics["smape"],
        metrics["wmape"],
        metrics["within_10pct"],
    )
    return metrics


def extract_feature_importances(
    model: RegressorMixin,
    feature_names: list[str],
) -> dict[str, float]:
    """Extract model-specific feature importance scores when available."""
    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(getattr(model, "feature_importances_"), dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(getattr(model, "coef_"), dtype=float)
        importances = np.abs(coefficients).reshape(-1)

    if importances is None or len(importances) != len(feature_names):
        return {}

    importance_dict = {
        feature_names[index]: float(importances[index]) for index in range(len(feature_names))
    }

    return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))


def prepare_data(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    x = df.drop(columns=[target_column])
    y = df[target_column]

    logger.info("Prepared data: %s samples, %s features", x.shape[0], x.shape[1])
    return x, y
