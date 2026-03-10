"""Competition workflow for baseline model selection."""

from __future__ import annotations

from cap3_classification.modeling.evaluation import fit_and_evaluate, split_xy
from cap3_classification.modeling.factory import ModelFactory
from cap3_classification.schemas import CompetitionSummary, ModelResult


def run_model_competition(
    train_df,
    validation_df,
    target_column: str,
    candidate_models: list[str],
    metric_name: str,
    random_state: int,
) -> CompetitionSummary:
    """Train naive baseline models and pick the best by the selected metric."""
    x_train, y_train = split_xy(train_df, target_column=target_column)
    x_validation, y_validation = split_xy(validation_df, target_column=target_column)

    results: list[ModelResult] = []
    for model_name in candidate_models:
        model = ModelFactory.create_candidate(model_name=model_name, random_state=random_state)
        metrics, elapsed = fit_and_evaluate(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_validation,
            y_eval=y_validation,
        )
        results.append(
            ModelResult(
                model_name=model_name,
                metrics=metrics,
                params={},
                training_time_seconds=elapsed,
            )
        )

    ordered = sorted(results, key=lambda item: item.metrics[metric_name], reverse=True)
    best = ordered[0]
    return CompetitionSummary(
        metric_name=metric_name,
        best_model_name=best.model_name,
        best_score=best.metrics[metric_name],
        candidate_models=candidate_models,
        results=results,
    )
