"""Tests for dataset loading."""

from __future__ import annotations

from cap3_classification.data import loaders


def test_loader_uses_digits_fallback():
    bundle = loaders.load_image_classification_dataset(
        primary_source="mnist",
        fallback_source="digits",
        fetch_strategy="fallback_only",
        max_samples=100,
        random_state=42,
        target_column="target",
    )
    assert len(bundle.dataframe) == 100
    assert bundle.metadata.source_used == "digits"
