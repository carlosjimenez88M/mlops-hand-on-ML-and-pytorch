"""
Utility functions for model selection
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time


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
        "DecisionTree": DecisionTreeRegressor(random_state=42)
    }
    return models


def get_default_param_grids() -> Dict[str, Dict[str, list]]:
    """
    Get default parameter grids for grid search.

    Returns:
        Dict with model name as key and param grid as value
    """
    param_grids = {
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        },
        "GradientBoosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "Ridge": {
            "alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "Lasso": {
            "alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "DecisionTree": {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        }
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
    Evaluate model with business-focused metrics.

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

    # Business-focused metrics
    mape = mean_absolute_percentage_error(y_true, y_pred)
    median_ape = median_absolute_percentage_error(y_true, y_pred)

    # Prediction accuracy at different thresholds
    within_5pct = predictions_within_threshold(y_true, y_pred, 0.05)
    within_10pct = predictions_within_threshold(y_true, y_pred, 0.10)
    within_15pct = predictions_within_threshold(y_true, y_pred, 0.15)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "median_ape": median_ape,
        "within_5pct": within_5pct,
        "within_10pct": within_10pct,
        "within_15pct": within_15pct
    }


def train_model_with_gridsearch(
    model: Any,
    param_grid: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 3
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Train model with GridSearchCV.

    Args:
        model: Model instance
        param_grid: Parameter grid for search
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds

    Returns:
        Tuple of (best_model, best_params, training_time)
    """
    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    training_time = time.time() - start_time

    return grid_search.best_estimator_, grid_search.best_params_, training_time
