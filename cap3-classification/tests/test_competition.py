"""Tests for model competition."""

from __future__ import annotations

from sklearn.datasets import load_digits

from cap3_classification.modeling.competition import run_model_competition


def test_model_competition_returns_best_model():
    dataset = load_digits(as_frame=True)
    dataframe = dataset.frame.rename(columns={"target": "target"})
    dataframe["target"] = dataframe["target"].astype(int)
    train_df = dataframe.iloc[:200].reset_index(drop=True)
    validation_df = dataframe.iloc[200:260].reset_index(drop=True)

    summary = run_model_competition(
        train_df=train_df,
        validation_df=validation_df,
        target_column="target",
        candidate_models=["dummy_most_frequent", "sgd_classifier"],
        metric_name="macro_f1",
        random_state=42,
    )

    assert summary.best_model_name in {"dummy_most_frequent", "sgd_classifier"}
    assert len(summary.results) == 2
