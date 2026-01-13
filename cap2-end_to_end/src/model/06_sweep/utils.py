"""
Utility functions for hyperparameter sweep.
"""
import io
import logging
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric MAPE (SMAPE).
    Less biased than MAPE toward underestimation.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


def weighted_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Weighted MAPE (wMAPE).
    Better for aggregate forecast accuracy.
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def predictions_within_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.10
) -> float:
    """Calculate % of predictions within threshold of actual."""
    errors = np.abs((y_pred / y_true) - 1)
    return (errors < threshold).mean() * 100


def download_data_from_gcs(bucket_name: str, gcs_path: str) -> pd.DataFrame:
    """
    Download data from GCS.

    Args:
        bucket_name: GCS bucket name
        gcs_path: Path to data in bucket

    Returns:
        DataFrame with data
    """
    logger.info(f"Downloading from GCS: gs://{bucket_name}/{gcs_path}")

    # Handle empty GOOGLE_APPLICATION_CREDENTIALS
    import os
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') == '':
        os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    content = blob.download_as_bytes()

    if gcs_path.endswith('.parquet'):
        df = pd.read_parquet(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content))

    logger.info(f"Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any]
) -> RandomForestRegressor:
    """
    Train Random Forest with given hyperparameters.

    Args:
        X_train: Training features
        y_train: Training target
        params: Hyperparameters

    Returns:
        Trained model
    """
    logger.info(f"Training Random Forest with params: {params}")

    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params.get('max_depth'),
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params.get('max_features', 'sqrt'),
        random_state=params.get('random_state', 42),
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    logger.info("Training completed")

    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model and return comprehensive metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target

    Returns:
        Dict with evaluation metrics including multiple percentage error metrics
    """
    y_pred = model.predict(X_test)
    y_true = y_test.values

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "smape": float(symmetric_mean_absolute_percentage_error(y_true, y_pred)),
        "wmape": float(weighted_mean_absolute_percentage_error(y_true, y_pred)),
        "median_ape": float(np.median(np.abs((y_true - y_pred) / y_true)) * 100),
        "within_5pct": float(predictions_within_threshold(y_true, y_pred, 0.05)),
        "within_10pct": float(predictions_within_threshold(y_true, y_pred, 0.10)),
        "within_15pct": float(predictions_within_threshold(y_true, y_pred, 0.15)),
    }

    logger.info(f"MAPE={metrics['mape']:.2f}% | SMAPE={metrics['smape']:.2f}% | "
                f"wMAPE={metrics['wmape']:.2f}% | Within10%={metrics['within_10pct']:.1f}%")

    return metrics


def log_feature_importances(model: RandomForestRegressor, feature_names: list) -> Dict[str, float]:
    """
    Extract and return feature importances from Random Forest model.

    Args:
        model: Trained Random Forest model
        feature_names: List of feature names

    Returns:
        Dict mapping feature names to importance scores
    """
    importances = model.feature_importances_
    feature_importance_dict = {
        feature_names[i]: float(importances[i])
        for i in range(len(feature_names))
    }

    # Sort by importance
    sorted_importances = dict(sorted(
        feature_importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    ))

    logger.info(f"\nTop 5 Most Important Features:")
    for i, (feature, importance) in enumerate(list(sorted_importances.items())[:5], 1):
        logger.info(f"  {i}. {feature}: {importance:.4f}")

    return sorted_importances


def prepare_data(
    df: pd.DataFrame,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target.

    Args:
        df: Input DataFrame
        target_column: Name of target column

    Returns:
        Tuple of (features, target)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")

    return X, y
