"""Tests for dataset splitting."""

from __future__ import annotations

from cap3_classification.data.splitters import split_dataset


def test_splitter_preserves_row_count(toy_dataframe):
    train_df, validation_df, test_df = split_dataset(
        dataframe=toy_dataframe,
        target_column="target",
        train_size=0.5,
        validation_size=0.25,
        test_size=0.25,
        random_state=42,
    )
    assert len(train_df) + len(validation_df) + len(test_df) == len(toy_dataframe)
