"""MLflow entrypoint for best-model hyperparameter tuning."""

from __future__ import annotations

import argparse

import mlflow

from cap3_classification.modeling.factory import ModelFactory
from cap3_classification.modeling.sweep import SweepContext, run_sweep
from cap3_classification.utils.io import read_json, read_parquet, write_yaml
from cap3_classification.utils.logging import get_logger
from cap3_classification.utils.paths import find_project_root, resolve_path

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--validation_path", required=True)
    parser.add_argument("--competition_summary_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--sweep_config_path", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--metric_name", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--offline_runs", required=True, type=int)
    parser.add_argument("--online_runs", required=True, type=int)
    parser.add_argument("--random_state", required=True, type=int)
    parser.add_argument("--target_column", required=True)
    parser.add_argument("--wandb_project", required=True)
    parser.add_argument("--wandb_mode", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = find_project_root()
    summary_path = resolve_path(args.competition_summary_path, root_path)
    output_path = resolve_path(args.output_path, root_path)
    sweep_config_path = resolve_path(args.sweep_config_path, root_path)
    train_path = resolve_path(args.train_path, root_path)
    validation_path = resolve_path(args.validation_path, root_path)

    summary = read_json(summary_path)
    best_model_name = summary["best_model_name"]
    search_space = ModelFactory.default_search_space(
        model_name=best_model_name,
        metric_name=args.metric_name,
        goal=args.goal,
    ).wandb

    with mlflow.start_run(run_name="06_sweep"):
        write_yaml(sweep_config_path, search_space)
        artifact = run_sweep(
            context=SweepContext(
                train_df=read_parquet(train_path),
                validation_df=read_parquet(validation_path),
                target_column=args.target_column,
                model_name=best_model_name,
                metric_name=args.metric_name,
                goal=args.goal,
                wandb_project=args.wandb_project,
                wandb_mode=args.wandb_mode,
                random_state=args.random_state,
            ),
            online_runs=args.online_runs,
            offline_runs=args.offline_runs,
        )

        write_yaml(output_path, artifact.model_dump(mode="json"))
        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_param("sweep_source", artifact.source)
        mlflow.log_metric(f"best_{artifact.metric_name}", artifact.best_score)
        mlflow.log_artifact(str(output_path))
        logger.info("Sweep completed for model %s", best_model_name)


if __name__ == "__main__":
    main()
