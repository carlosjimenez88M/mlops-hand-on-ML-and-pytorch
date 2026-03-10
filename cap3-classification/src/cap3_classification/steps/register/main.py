"""MLflow entrypoint for final model registration."""

from __future__ import annotations

import argparse
import os

import mlflow
import wandb

from cap3_classification.modeling.registration import register_model
from cap3_classification.utils.io import read_parquet, read_yaml, write_json
from cap3_classification.utils.logging import get_logger
from cap3_classification.utils.paths import find_project_root, resolve_path

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--validation_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--best_params_path", required=True)
    parser.add_argument("--registered_model_name", required=True)
    parser.add_argument("--model_stage", required=True)
    parser.add_argument("--target_column", required=True)
    parser.add_argument("--local_model_output_path", required=True)
    parser.add_argument("--serving_metadata_path", required=True)
    parser.add_argument("--wandb_project", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = find_project_root()
    train_path = resolve_path(args.train_path, root_path)
    validation_path = resolve_path(args.validation_path, root_path)
    test_path = resolve_path(args.test_path, root_path)
    best_params_path = resolve_path(args.best_params_path, root_path)
    local_model_output_path = resolve_path(args.local_model_output_path, root_path)
    serving_metadata_path = resolve_path(args.serving_metadata_path, root_path)

    best_params_payload = read_yaml(best_params_path)
    run = wandb.init(
        project=args.wandb_project,
        job_type="registration",
        mode=os.getenv("WANDB_MODE", "offline"),
    )
    with mlflow.start_run(run_name="07_register"):
        artifact = register_model(
            train_df=read_parquet(train_path),
            validation_df=read_parquet(validation_path),
            test_df=read_parquet(test_path),
            target_column=args.target_column,
            selected_model_name=best_params_payload["model_name"],
            best_params=best_params_payload["best_params"],
            registered_model_name=args.registered_model_name,
            model_stage=args.model_stage,
            local_model_output_path=local_model_output_path,
        )
        write_json(serving_metadata_path, artifact.model_dump(mode="json"))
        mlflow.log_artifact(str(serving_metadata_path))
        wandb.log({"model_version": artifact.model_version, **artifact.metrics})
        run.finish()
        logger.info("Registered model version %s", artifact.model_version)


if __name__ == "__main__":
    main()
