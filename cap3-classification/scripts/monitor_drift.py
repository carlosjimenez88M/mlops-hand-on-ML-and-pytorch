"""Generate a PSI drift report from two local parquet datasets."""

from __future__ import annotations

import argparse

from cap3_classification.monitoring.drift import generate_drift_report
from cap3_classification.utils.io import read_parquet, write_json
from cap3_classification.utils.paths import find_project_root, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", default="data/04_split/train.parquet")
    parser.add_argument("--current_path", default="data/04_split/test.parquet")
    parser.add_argument("--report_path", default="reports/drift/drift_report.json")
    parser.add_argument("--max_psi", default=0.2, type=float)
    parser.add_argument("--target_column", default="target")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = find_project_root()
    report = generate_drift_report(
        reference_df=read_parquet(resolve_path(args.reference_path, root_path)),
        current_df=read_parquet(resolve_path(args.current_path, root_path)),
        target_column=args.target_column,
        max_psi=args.max_psi,
    )
    write_json(resolve_path(args.report_path, root_path), report.model_dump(mode="json"))


if __name__ == "__main__":
    main()
