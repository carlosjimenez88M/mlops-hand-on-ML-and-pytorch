"""Promote an MLflow model version from one stage to another with metric guardrails."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
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


class PromotionConfig(BaseModel):
    """Runtime configuration for model stage promotion."""

    model_name: str = Field(..., min_length=1)
    from_stage: str = Field(default="Staging", min_length=1)
    to_stage: str = Field(default="Production", min_length=1)
    min_r2: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    max_mape: Optional[float] = Field(default=None, ge=0.0)
    archive_existing_production: bool = True
    report_path: Path = Path("artifacts/promotion/promotion_report.json")

    @field_validator("model_name", "from_stage", "to_stage")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value cannot be empty")
        return stripped


class PromotionResult(BaseModel):
    """Promotion execution summary."""

    promoted: bool
    model_name: str
    from_stage: str
    to_stage: str
    candidate_version: str
    reason: str
    r2: Optional[float] = None
    mape: Optional[float] = None
    promoted_at: str


class ModelPromotionWorkflow:
    """Encapsulates MLflow registry promotion workflow."""

    def __init__(self, config: PromotionConfig):
        self.config = config
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            tracking_uri = f"sqlite:///{Path('mlflow.db').resolve()}"
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

        mlflow.set_tracking_uri(tracking_uri)
        logger.info("Using MLflow tracking URI: %s", tracking_uri)
        self.client = MlflowClient()

    @staticmethod
    def _normalize_alias(label: str) -> str:
        """Normalize human stage label into MLflow alias format."""
        return label.strip().lower().replace(" ", "_")

    def _get_candidate_version(self) -> str:
        """Resolve candidate version from source alias/stage label."""
        source_alias = self._normalize_alias(self.config.from_stage)

        try:
            aliased_version = self.client.get_model_version_by_alias(
                name=self.config.model_name,
                alias=source_alias,
            )
        except MlflowException as error:
            # Fallback: search by tag for backward compatibility.
            try:
                versions = self.client.search_model_versions(f"name='{self.config.model_name}'")
            except MlflowException as search_error:
                raise RuntimeError(
                    f"Could not read model '{self.config.model_name}' from MLflow registry: "
                    f"{search_error}"
                ) from search_error

            matching = [
                version
                for version in versions
                if (version.tags or {}).get("deployment_alias") == source_alias
            ]
            if not matching:
                raise RuntimeError(
                    f"No version found with alias/stage label '{self.config.from_stage}' "
                    f"(normalized alias '{source_alias}') for model '{self.config.model_name}'. "
                    f"Original error: {error}"
                ) from error

            latest = max(matching, key=lambda version: int(version.version))
            return str(latest.version)

        return str(aliased_version.version)

    def _read_metric_tags(self, version: str) -> tuple[Optional[float], Optional[float]]:
        """Extract r2/mape metric tags from model version metadata."""
        model_version = self.client.get_model_version(name=self.config.model_name, version=version)
        tags = model_version.tags or {}

        def parse_float(value: Optional[str]) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        return parse_float(tags.get("r2")), parse_float(tags.get("mape"))

    def _validate_guardrails(
        self, version: str
    ) -> tuple[bool, str, Optional[float], Optional[float]]:
        """Validate promotion thresholds against version tags."""
        r2, mape = self._read_metric_tags(version)

        if self.config.min_r2 is not None:
            if r2 is None:
                return False, "Missing r2 tag for guardrail validation", r2, mape
            if r2 < self.config.min_r2:
                return (
                    False,
                    f"r2 guardrail failed ({r2:.4f} < {self.config.min_r2:.4f})",
                    r2,
                    mape,
                )

        if self.config.max_mape is not None:
            if mape is None:
                return False, "Missing mape tag for guardrail validation", r2, mape
            if mape > self.config.max_mape:
                return (
                    False,
                    f"mape guardrail failed ({mape:.4f} > {self.config.max_mape:.4f})",
                    r2,
                    mape,
                )

        return True, "Guardrails passed", r2, mape

    def _promote(self, version: str) -> None:
        """Assign destination alias to selected model version."""
        destination_alias = self._normalize_alias(self.config.to_stage)
        self.client.set_registered_model_alias(
            name=self.config.model_name,
            alias=destination_alias,
            version=version,
        )
        self.client.set_model_version_tag(
            name=self.config.model_name,
            version=version,
            key="deployment_alias",
            value=destination_alias,
        )

    def run(self) -> PromotionResult:
        """Execute guarded promotion and return structured result."""
        logger.info("Starting model promotion workflow for '%s'", self.config.model_name)
        candidate_version = self._get_candidate_version()
        logger.info(
            "Candidate version resolved from '%s' is v%s",
            self.config.from_stage,
            candidate_version,
        )

        passed, reason, r2, mape = self._validate_guardrails(candidate_version)
        if not passed:
            return PromotionResult(
                promoted=False,
                model_name=self.config.model_name,
                from_stage=self.config.from_stage,
                to_stage=self.config.to_stage,
                candidate_version=candidate_version,
                reason=reason,
                r2=r2,
                mape=mape,
                promoted_at=datetime.now(timezone.utc).isoformat(),
            )

        self._promote(candidate_version)
        logger.info(
            "Version v%s promoted from '%s' to '%s'",
            candidate_version,
            self.config.from_stage,
            self.config.to_stage,
        )
        return PromotionResult(
            promoted=True,
            model_name=self.config.model_name,
            from_stage=self.config.from_stage,
            to_stage=self.config.to_stage,
            candidate_version=candidate_version,
            reason="Promotion completed successfully",
            r2=r2,
            mape=mape,
            promoted_at=datetime.now(timezone.utc).isoformat(),
        )


def parse_args() -> PromotionConfig:
    """Parse CLI args into validated pydantic config."""
    parser = argparse.ArgumentParser(description="Promote an MLflow model version.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--from-stage", type=str, default="Staging")
    parser.add_argument("--to-stage", type=str, default="Production")
    parser.add_argument("--min-r2", type=float, default=None)
    parser.add_argument("--max-mape", type=float, default=None)
    parser.add_argument(
        "--archive-existing-production",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/promotion/promotion_report.json"),
    )
    args = parser.parse_args()
    return PromotionConfig(**vars(args))


def write_report(report_path: Path, result: PromotionResult) -> None:
    """Persist promotion result to JSON for CI artifacts."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    logger.info("Promotion report written to %s", report_path)


def main() -> None:
    """CLI entrypoint."""
    config = parse_args()
    try:
        workflow = ModelPromotionWorkflow(config=config)
        result = workflow.run()
    except Exception as error:
        result = PromotionResult(
            promoted=False,
            model_name=config.model_name,
            from_stage=config.from_stage,
            to_stage=config.to_stage,
            candidate_version="unknown",
            reason=str(error),
            promoted_at=datetime.now(timezone.utc).isoformat(),
        )

    write_report(config.report_path, result)

    if not result.promoted:
        logger.error("Promotion skipped: %s", result.reason)
        raise SystemExit(1)

    logger.info("Promotion completed: %s", result.reason)


if __name__ == "__main__":
    main()
