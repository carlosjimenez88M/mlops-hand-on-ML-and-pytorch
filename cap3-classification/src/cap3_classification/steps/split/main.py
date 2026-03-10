"""MLflow entrypoint for train, validation and test split."""

from __future__ import annotations

import argparse

import mlflow

from cap3_classification.data.splitters import split_dataset
from cap3_classification.utils.io import read_parquet, write_parquet
from cap3_classification.utils.logging import get_logger
from cap3_classification.utils.paths import find_project_root, resolve_path

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--validation_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--train_size", required=True, type=float)
    parser.add_argument("--validation_size", required=True, type=float)
    parser.add_argument("--test_size", required=True, type=float)
    parser.add_argument("--random_state", required=True, type=int)
    parser.add_argument("--target_column", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = find_project_root()
    input_path = resolve_path(args.input_path, root_path)
    train_path = resolve_path(args.train_path, root_path)
    validation_path = resolve_path(args.validation_path, root_path)
    test_path = resolve_path(args.test_path, root_path)

    with mlflow.start_run(run_name="04_split"):
        dataframe = read_parquet(input_path)
        train_df, validation_df, test_df = split_dataset(
            dataframe=dataframe,
            target_column=args.target_column,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        write_parquet(train_path, train_df)
        write_parquet(validation_path, validation_df)
        write_parquet(test_path, test_df)
        mlflow.log_metrics(
            {
                "train_rows": len(train_df),
                "validation_rows": len(validation_df),
                "test_rows": len(test_df),
            }
        )
        logger.info("Split completed")


if __name__ == "__main__":
    main()
