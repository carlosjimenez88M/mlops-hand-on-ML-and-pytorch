"""Simple PSI-based numeric data drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cap3_classification.schemas import DriftFeatureResult, DriftReport


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute PSI for a single numeric feature."""
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.quantile(expected, quantiles)
    cut_points = np.unique(cut_points)
    if len(cut_points) < 3:
        min_value = float(np.min(expected))
        max_value = float(np.max(expected))
        spread = max(max_value - min_value, 1.0)
        cut_points = np.array([min_value - spread, min_value, max_value + spread], dtype=float)
    cut_points[0] = float("-inf")
    cut_points[-1] = float("inf")

    expected_hist, _ = np.histogram(expected, bins=cut_points)
    actual_hist, _ = np.histogram(actual, bins=cut_points)

    expected_dist = np.where(expected_hist == 0, 1e-6, expected_hist / max(expected_hist.sum(), 1))
    actual_dist = np.where(actual_hist == 0, 1e-6, actual_hist / max(actual_hist.sum(), 1))
    return float(np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)))


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: str,
    max_psi: float,
) -> DriftReport:
    """Compute PSI across shared numeric features."""
    features = [column for column in reference_df.columns if column != target_column]
    results: list[DriftFeatureResult] = []
    for feature in features:
        psi = population_stability_index(
            expected=reference_df[feature].to_numpy(),
            actual=current_df[feature].to_numpy(),
        )
        results.append(DriftFeatureResult(feature=feature, psi=psi))

    ordered = sorted(results, key=lambda item: item.psi, reverse=True)
    drifted = [item for item in ordered if item.psi > max_psi]
    return DriftReport(
        drift_detected=bool(drifted),
        max_psi_threshold=max_psi,
        drifted_feature_count=len(drifted),
        top_features_by_psi=ordered[:10],
    )
