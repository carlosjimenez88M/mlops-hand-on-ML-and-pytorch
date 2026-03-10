"""MLflow entrypoint for baseline model competition."""

from __future__ import annotations

import argparse
import os

import mlflow
import wandb

from cap3_classification.modeling.competition import run_model_competition
from cap3_classification.utils.io import read_parquet, write_json
from cap3_classification.utils.logging import get_logger
from cap3_classification.utils.paths import find_project_root, resolve_path

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--validation_path", required=True)
    parser.add_argument("--target_column", required=True)
    parser.add_argument("--metric_name", required=True)
    parser.add_argument("--results_path", required=True)
    parser.add_argument("--best_model_summary_path", required=True)
    parser.add_argument("--candidate_models", required=True)
    parser.add_argument("--random_state", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = find_project_root()
    train_path = resolve_path(args.train_path, root_path)
    validation_path = resolve_path(args.validation_path, root_path)
    results_path = resolve_path(args.results_path, root_path)
    summary_path = resolve_path(args.best_model_summary_path, root_path)
    candidate_models = [item.strip() for item in args.candidate_models.split(",") if item.strip()]

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "cap3-classification"),
        job_type="model_competition",
        mode=os.getenv("WANDB_MODE", "offline"),
        config=vars(args),
    )
    with mlflow.start_run(run_name="05_model_competition"):
        summary = run_model_competition(
            train_df=read_parquet(train_path),
            validation_df=read_parquet(validation_path),
            target_column=args.target_column,
            candidate_models=candidate_models,
            metric_name=args.metric_name,
            random_state=args.random_state,
        )
        payload = summary.model_dump(mode="json")
        write_json(results_path, payload)
        write_json(summary_path, payload)

        mlflow.log_param("best_model_name", summary.best_model_name)
        mlflow.log_metric(f"best_{summary.metric_name}", summary.best_score)
        mlflow.log_artifact(str(results_path))
        wandb.log(
            {
                "best_model_name": summary.best_model_name,
                f"best_{summary.metric_name}": summary.best_score,
            }
        )
        run.finish()
        logger.info("Best model selected: %s", summary.best_model_name)


if __name__ == "__main__":
    main()
