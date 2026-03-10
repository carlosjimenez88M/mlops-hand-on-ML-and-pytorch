"""MLflow entrypoint for dataset validation."""

from __future__ import annotations

import argparse

import mlflow

from cap3_classification.data.validators import validate_dataset
from cap3_classification.utils.io import read_parquet, write_json, write_parquet
from cap3_classification.utils.logging import get_logger
from cap3_classification.utils.paths import find_project_root, resolve_path

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--min_rows", required=True, type=int)
    parser.add_argument("--min_features", required=True, type=int)
    parser.add_argument("--target_column", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = find_project_root()
    input_path = resolve_path(args.input_path, root_path)
    output_path = resolve_path(args.output_path, root_path)
    report_path = resolve_path(args.report_path, root_path)

    with mlflow.start_run(run_name="02_validate"):
        dataframe = read_parquet(input_path)
        report = validate_dataset(
            dataframe=dataframe,
            target_column=args.target_column,
            min_rows=args.min_rows,
            min_features=args.min_features,
        )
        write_parquet(output_path, dataframe)
        write_json(report_path, report.model_dump(mode="json"))

        mlflow.log_metrics(
            {
                "row_count": report.row_count,
                "feature_count": report.feature_count,
                "class_count": report.class_count,
                "missing_values": report.missing_values,
                "duplicate_rows": report.duplicate_rows,
            }
        )
        mlflow.log_artifact(str(report_path))
        logger.info("Validation completed with status=%s", report.status)


if __name__ == "__main__":
    main()
