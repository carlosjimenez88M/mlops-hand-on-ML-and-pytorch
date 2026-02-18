"""Compute tabular data drift with PSI and emit a machine-readable report."""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from google.cloud import storage
from pydantic import BaseModel, Field, field_validator

try:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.utils.colored_logger import setup_colored_logger

    setup_colored_logger()
    logger = logging.getLogger(__name__)
except (ImportError, Exception):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)


class DriftMonitorConfig(BaseModel):
    """Validated runtime arguments for drift checks."""

    bucket_name: str = Field(..., min_length=1)
    reference_path: str = Field(..., min_length=1)
    current_path: str = Field(..., min_length=1)
    target_column: str | None = "median_house_value"
    bins: int = Field(default=10, ge=5, le=50)
    max_psi: float = Field(default=0.2, ge=0.0)
    min_rows: int = Field(default=100, ge=1)
    report_path: Path = Path("artifacts/monitoring/drift_report.json")
    fail_on_drift: bool = False

    @field_validator("bucket_name", "reference_path", "current_path")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value cannot be empty")
        return stripped


class GCSParquetRepository:
    """Simple GCS repository for loading parquet datasets."""

    def __init__(self, bucket_name: str):
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._bucket_name = bucket_name

    def load_dataframe(self, gcs_path: str) -> pd.DataFrame:
        """Download parquet blob from GCS and return dataframe."""
        blob = self._bucket.blob(gcs_path)
        if not blob.exists(self._client):
            raise FileNotFoundError(f"Blob not found: gs://{self._bucket_name}/{gcs_path}")
        content = blob.download_as_bytes()
        return pd.read_parquet(io.BytesIO(content))


class PopulationStabilityIndex:
    """PSI calculator for a pair of numeric distributions."""

    EPSILON = 1e-6

    @classmethod
    def calculate(cls, reference: pd.Series, current: pd.Series, bins: int) -> float:
        """Calculate PSI for two numeric series."""
        ref = reference.dropna().astype(float)
        cur = current.dropna().astype(float)
        if ref.empty or cur.empty:
            return 0.0

        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(ref, quantiles)
        edges = np.unique(edges)
        if len(edges) < 2:
            return 0.0

        ref_bins = pd.cut(ref, bins=edges, include_lowest=True, duplicates="drop")
        cur_bins = pd.cut(cur, bins=edges, include_lowest=True, duplicates="drop")

        ref_dist = ref_bins.value_counts(normalize=True, sort=False)
        cur_dist = cur_bins.value_counts(normalize=True, sort=False).reindex(
            ref_dist.index, fill_value=0.0
        )

        ref_values = ref_dist.to_numpy() + cls.EPSILON
        cur_values = cur_dist.to_numpy() + cls.EPSILON

        psi = np.sum((cur_values - ref_values) * np.log(cur_values / ref_values))
        return float(psi)


class DriftMonitorWorkflow:
    """Orchestrates data drift checks and report generation."""

    def __init__(self, config: DriftMonitorConfig):
        self.config = config
        self.repository = GCSParquetRepository(bucket_name=config.bucket_name)

    def _numeric_features(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> list[str]:
        """Return numeric feature intersection excluding optional target column."""
        reference_numeric = set(reference_df.select_dtypes(include=["number"]).columns.tolist())
        current_numeric = set(current_df.select_dtypes(include=["number"]).columns.tolist())
        common = sorted(reference_numeric.intersection(current_numeric))
        if self.config.target_column in common:
            common.remove(self.config.target_column)
        return common

    def _build_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        psi_by_feature: dict[str, float],
    ) -> dict[str, Any]:
        """Build structured drift report."""
        drifted = {
            feature: value
            for feature, value in psi_by_feature.items()
            if value >= self.config.max_psi
        }
        sorted_psi = sorted(psi_by_feature.items(), key=lambda item: item[1], reverse=True)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bucket_name": self.config.bucket_name,
            "reference_path": self.config.reference_path,
            "current_path": self.config.current_path,
            "reference_rows": int(len(reference_df)),
            "current_rows": int(len(current_df)),
            "max_psi_threshold": float(self.config.max_psi),
            "drift_detected": bool(drifted),
            "drifted_feature_count": len(drifted),
            "drifted_features": drifted,
            "top_features_by_psi": sorted_psi[:10],
        }

    def run(self) -> dict[str, Any]:
        """Execute drift checks and return report payload."""
        logger.info(
            "Loading reference dataset from gs://%s/%s",
            self.config.bucket_name,
            self.config.reference_path,
        )
        reference_df = self.repository.load_dataframe(self.config.reference_path)
        logger.info(
            "Loading current dataset from gs://%s/%s",
            self.config.bucket_name,
            self.config.current_path,
        )
        current_df = self.repository.load_dataframe(self.config.current_path)

        if len(reference_df) < self.config.min_rows:
            raise ValueError(
                f"Reference dataset too small ({len(reference_df)} rows < {self.config.min_rows})"
            )
        if len(current_df) < self.config.min_rows:
            raise ValueError(
                f"Current dataset too small ({len(current_df)} rows < {self.config.min_rows})"
            )

        features = self._numeric_features(reference_df=reference_df, current_df=current_df)
        if not features:
            raise ValueError("No common numeric features found to compute drift.")

        psi_by_feature: dict[str, float] = {}
        for feature in features:
            psi_by_feature[feature] = PopulationStabilityIndex.calculate(
                reference=reference_df[feature],
                current=current_df[feature],
                bins=self.config.bins,
            )

        report = self._build_report(
            reference_df=reference_df,
            current_df=current_df,
            psi_by_feature=psi_by_feature,
        )
        return report


def parse_args() -> DriftMonitorConfig:
    """Parse CLI args into validated config."""
    parser = argparse.ArgumentParser(description="Run PSI-based data drift monitoring.")
    parser.add_argument("--bucket-name", type=str, required=True)
    parser.add_argument("--reference-path", type=str, required=True)
    parser.add_argument("--current-path", type=str, required=True)
    parser.add_argument("--target-column", type=str, default="median_house_value")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max-psi", type=float, default=0.2)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/monitoring/drift_report.json"),
    )
    parser.add_argument(
        "--fail-on-drift",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    return DriftMonitorConfig(**vars(args))


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Persist drift report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Drift report saved to %s", path)


def main() -> None:
    """CLI entrypoint."""
    config = parse_args()
    workflow = DriftMonitorWorkflow(config=config)
    report = workflow.run()
    write_report(config.report_path, report)

    logger.info(
        "Drift check completed. drift_detected=%s, drifted_feature_count=%s",
        report["drift_detected"],
        report["drifted_feature_count"],
    )
    if report["drift_detected"] and config.fail_on_drift:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
