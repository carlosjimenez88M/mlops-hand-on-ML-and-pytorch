"""
Utility functions for model selection
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


def get_available_models() -> Dict[str, Any]:
    """
    Get dictionary of available regression models.

    Returns:
        Dict with model name as key and model instance as value
    """
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
    }
    return models


def get_default_param_grids() -> Dict[str, Dict[str, list]]:
    """
    Get refined parameter grids for grid search based on domain knowledge.

    Improvements:
    - Ridge/Lasso: More alpha values in logarithmic scale for better regularization tuning
    - GradientBoosting: Extended learning_rate range and subsample parameter
    - RandomForest: Added min_samples_leaf and more granular depth options
    - DecisionTree: Added min_samples_leaf for better overfitting control

    Returns:
        Dict with model name as key and param grid as value
    """
    param_grids = {
        "RandomForest": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "GradientBoosting": {
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "max_depth": [3, 4, 5, 6, 7],
            "subsample": [0.8, 0.9, 1.0],
        },
        "Ridge": {
            "alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
        },
        "Lasso": {
            "alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
        },
        "DecisionTree": {
            "max_depth": [5, 10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
        },
    }
    return param_grids


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Business interpretation: Average percentage error in predictions.
    Lower is better. 5% means predictions are off by 5% on average.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAPE as percentage (0-100)
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def median_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Median Absolute Percentage Error.
    More robust to outliers than MAPE.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Median APE as percentage (0-100)
    """
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric MAPE (SMAPE).

    SMAPE addresses MAPE's bias toward low values by using the average of
    actual and predicted in the denominator. Range: 0-200%.

    Advantages over MAPE:
    - Less biased toward underestimation
    - More symmetric treatment of over/under predictions
    - Better for wide value ranges

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        SMAPE as percentage (0-200)
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


def weighted_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Weighted MAPE (wMAPE).

    wMAPE uses the sum of actual values as denominator, making it less sensitive
    to individual extreme values and more representative of overall error.

    Advantages:
    - Not affected by individual zero or near-zero values
    - Better represents aggregate forecast accuracy
    - Industry standard for demand forecasting

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        wMAPE as percentage (0-100+)
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def predictions_within_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """
    Calculate percentage of predictions within threshold of actual values.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        threshold: Error threshold (e.g., 0.05 for 5%)

    Returns:
        Percentage of predictions within threshold (0-100)
    """
    err = np.abs((y_pred - y_true) / y_true)
    within_threshold = err < threshold
    return within_threshold.mean() * 100


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model with comprehensive business-focused metrics.

    Metrics include traditional regression metrics (MAE, RMSE, R²) and
    multiple percentage error metrics to handle MAPE bias issues:
    - MAPE: Standard but biased toward low values
    - SMAPE: Symmetric, less biased
    - wMAPE: Weighted, better for aggregate accuracy
    - Median APE: Robust to outliers

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target

    Returns:
        Dict with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_true = y_test.values

    # Traditional metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Business-focused percentage error metrics (handling MAPE bias)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    wmape = weighted_mean_absolute_percentage_error(y_true, y_pred)
    median_ape = median_absolute_percentage_error(y_true, y_pred)

    # Prediction accuracy at different thresholds
    within_5pct = predictions_within_threshold(y_true, y_pred, 0.05)
    within_10pct = predictions_within_threshold(y_true, y_pred, 0.10)
    within_15pct = predictions_within_threshold(y_true, y_pred, 0.15)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "smape": float(smape),
        "wmape": float(wmape),
        "median_ape": float(median_ape),
        "within_5pct": float(within_5pct),
        "within_10pct": float(within_10pct),
        "within_15pct": float(within_15pct),
    }


def train_model_with_gridsearch(
    model: Any, param_grid: Dict[str, list], X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5
) -> Tuple[Any, Dict[str, Any], float, Dict[str, float]]:
    """
    Train model with K-fold Cross-Validation via GridSearchCV.

    Improvements:
    - Uses cv=5 (default) for more robust evaluation on small/medium datasets
    - Returns CV scores to understand model stability
    - Uses negative MAE as scoring metric (more robust than R² for this domain)

    Args:
        model: Model instance
        param_grid: Parameter grid for search
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds (default: 5 for better estimates)

    Returns:
        Tuple of (best_model, best_params, training_time, cv_scores)
    """
    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Extract cross-validation results
    cv_metrics = {
        "mean_test_score": float(-grid_search.best_score_),
        "std_test_score": float(grid_search.cv_results_["std_test_score"][grid_search.best_index_]),
        "mean_train_score": float(
            -grid_search.cv_results_["mean_train_score"][grid_search.best_index_]
        ),
        "std_train_score": float(
            grid_search.cv_results_["std_train_score"][grid_search.best_index_]
        ),
    }

    return grid_search.best_estimator_, grid_search.best_params_, training_time, cv_metrics
