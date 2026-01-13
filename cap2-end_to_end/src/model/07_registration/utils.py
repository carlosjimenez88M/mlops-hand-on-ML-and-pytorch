"""
Utility functions for model registration.
"""
import io
import logging
import pickle
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE."""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric MAPE (SMAPE) - less biased than MAPE."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def weighted_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Weighted MAPE (wMAPE) - better for aggregate accuracy."""
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)


def predictions_within_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.10
) -> float:
    """Calculate % of predictions within threshold of actual."""
    errors = np.abs((y_pred / y_true) - 1)
    return float((errors < threshold).mean() * 100)


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


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any]
) -> RandomForestRegressor:
    """
    Train final Random Forest model with optimized hyperparameters.

    Args:
        X_train: Training features
        y_train: Training target
        params: Optimized hyperparameters

    Returns:
        Trained model
    """
    logger.info("=" * 70)
    logger.info("TRAINING FINAL MODEL")
    logger.info("=" * 70)
    logger.info(f"Hyperparameters: {params}")

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
    logger.info(" Training completed")

    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model and return comprehensive metrics including multiple percentage error metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target

    Returns:
        Dict with evaluation metrics
    """
    logger.info("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_true = y_test.values

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        "wmape": weighted_mean_absolute_percentage_error(y_true, y_pred),
        "median_ape": float(np.median(np.abs((y_true - y_pred) / y_true)) * 100),
        "within_5pct": predictions_within_threshold(y_true, y_pred, 0.05),
        "within_10pct": predictions_within_threshold(y_true, y_pred, 0.10),
        "within_15pct": predictions_within_threshold(y_true, y_pred, 0.15),
    }

    logger.info("\nFinal Model Metrics:")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  SMAPE: {metrics['smape']:.2f}%")
    logger.info(f"  wMAPE: {metrics['wmape']:.2f}%")
    logger.info(f"  Median APE: {metrics['median_ape']:.2f}%")
    logger.info(f"  Within 10%: {metrics['within_10pct']:.1f}%")
    logger.info(f"  RMSE: {metrics['rmse']:.2f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")

    return metrics


def save_model_locally(model: RandomForestRegressor, output_path: Path) -> None:
    """
    Save model to local file.

    Args:
        model: Trained model
        output_path: Path to save model
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f" Model saved to: {output_path}")
