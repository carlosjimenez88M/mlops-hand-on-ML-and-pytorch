"""MLflow entrypoint for dataset ingestion."""

from __future__ import annotations

import argparse
import os

import mlflow
import wandb

from cap3_classification.data.loaders import load_image_classification_dataset
from cap3_classification.tracking.wandb_utils import init_run
from cap3_classification.utils.io import write_json, write_parquet
from cap3_classification.utils.logging import get_logger
from cap3_classification.utils.paths import find_project_root, resolve_path

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary_source", required=True)
    parser.add_argument("--fallback_source", required=True)
    parser.add_argument("--fetch_strategy", required=True)
    parser.add_argument("--max_samples", required=True, type=int)
    parser.add_argument("--random_state", required=True, type=int)
    parser.add_argument("--target_column", required=True)
    parser.add_argument("--raw_dataset_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = find_project_root()
    dataset_path = resolve_path(args.raw_dataset_path, root_path)
    metadata_path = resolve_path(args.metadata_path, root_path)

    run = init_run(
        project=os.getenv("WANDB_PROJECT", "cap3-classification"),
        job_type="ingest",
        mode=os.getenv("WANDB_MODE", "offline"),
        config=vars(args),
    )
    with mlflow.start_run(run_name="01_ingest"):
        bundle = load_image_classification_dataset(
            primary_source=args.primary_source,
            fallback_source=args.fallback_source,
            fetch_strategy=args.fetch_strategy,
            max_samples=args.max_samples,
            random_state=args.random_state,
            target_column=args.target_column,
        )
        write_parquet(dataset_path, bundle.dataframe)
        write_json(metadata_path, bundle.metadata.model_dump(mode="json"))

        mlflow.log_param("source_requested", args.primary_source)
        mlflow.log_param("source_used", bundle.metadata.source_used)
        mlflow.log_metric("rows", bundle.metadata.n_samples)
        mlflow.log_metric("features", bundle.metadata.n_features)
        mlflow.log_artifact(str(metadata_path))

        if run is not None:
            wandb.log({"rows": bundle.metadata.n_samples, "features": bundle.metadata.n_features})
            run.finish()

        logger.info("Saved raw dataset to %s", dataset_path)


if __name__ == "__main__":
    main()
