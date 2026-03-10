"""Tests for feature engineering."""

from __future__ import annotations

from cap3_classification.data.features import build_features


def test_feature_engineering_adds_derived_columns(toy_dataframe):
    engineered, manifest = build_features(
        dataframe=toy_dataframe,
        target_column="target",
        activation_threshold=0.5,
    )
    assert "pixel_mean" in engineered.columns
    assert manifest.total_feature_count > manifest.base_feature_count
