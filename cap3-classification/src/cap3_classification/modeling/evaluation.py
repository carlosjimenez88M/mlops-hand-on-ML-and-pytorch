"""Model evaluation helpers for multi-class classification."""

from __future__ import annotations

from time import perf_counter

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def split_xy(dataframe: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split features and target from a dataframe."""
    return dataframe.drop(columns=[target_column]), dataframe[target_column]


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    """Compute the metric suite used for competition and registration."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def fit_and_evaluate(model, x_train, y_train, x_eval, y_eval) -> tuple[dict[str, float], float]:
    """Train the model and evaluate it, returning metrics and fit time."""
    start = perf_counter()
    model.fit(x_train, y_train)
    elapsed = perf_counter() - start
    predictions = model.predict(x_eval)
    metrics = evaluate_predictions(y_true=y_eval, y_pred=predictions)
    return metrics, elapsed
