"""MLflow entrypoint for feature engineering."""

from __future__ import annotations

import argparse

import mlflow

from cap3_classification.data.features import build_features
from cap3_classification.utils.io import read_parquet, write_json, write_parquet
from cap3_classification.utils.logging import get_logger
from cap3_classification.utils.paths import find_project_root, resolve_path

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--activation_threshold", required=True, type=float)
    parser.add_argument("--target_column", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = find_project_root()
    input_path = resolve_path(args.input_path, root_path)
    output_path = resolve_path(args.output_path, root_path)
    metadata_path = resolve_path(args.metadata_path, root_path)

    with mlflow.start_run(run_name="03_feature_engineering"):
        dataframe = read_parquet(input_path)
        engineered, manifest = build_features(
            dataframe=dataframe,
            target_column=args.target_column,
            activation_threshold=args.activation_threshold,
        )
        write_parquet(output_path, engineered)
        write_json(metadata_path, manifest.model_dump(mode="json"))
        mlflow.log_metric("total_features", manifest.total_feature_count)
        mlflow.log_metric("derived_features", len(manifest.derived_feature_names))
        mlflow.log_artifact(str(metadata_path))
        logger.info("Engineered dataset saved to %s", output_path)


if __name__ == "__main__":
    main()
