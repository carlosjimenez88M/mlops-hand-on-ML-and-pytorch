"""W&B Bayesian sweep step for post-selection model fine-tuning."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

import wandb
from models import SupportedModelType, SweepBestParams, SweepExecutionConfig, SweepRunSummary
from utils import (
    GCSDataRepository,
    ModelFactory,
    evaluate_model,
    extract_feature_importances,
    prepare_data,
    train_model,
)

try:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.utils.colored_logger import setup_colored_logger

    setup_colored_logger()
    logger = logging.getLogger(__name__)
except (ImportError, Exception):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)


class BayesianSweepWorkflow:
    """Orchestrates model-specific Bayesian sweep execution."""

    def __init__(self, config: SweepExecutionConfig):
        self.config = config
        self.repository = GCSDataRepository(bucket_name=config.bucket_name)

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.feature_names: list[str] = []

        self.run_summaries: list[SweepRunSummary] = []

    def _load_data_once(self) -> None:
        """Load and prepare train/test data a single time for all sweep runs."""
        logger.info("Loading training data from GCS...")
        train_df = self.repository.download_dataframe(self.config.gcs_train_path)
        self.x_train, self.y_train = prepare_data(train_df, self.config.target_column)

        logger.info("Loading test data from GCS...")
        test_df = self.repository.download_dataframe(self.config.gcs_test_path)
        self.x_test, self.y_test = prepare_data(test_df, self.config.target_column)

        self.feature_names = self.x_train.columns.tolist()

        logger.info("Data cache ready")
        logger.info("  Train: %s", self.x_train.shape)
        logger.info("  Test: %s", self.x_test.shape)
        logger.info("  Features: %s", len(self.feature_names))

    def _load_custom_random_forest_config(self) -> dict[str, Any] | None:
        """Load custom RF sweep config when available."""
        if self.config.best_model_type != SupportedModelType.RANDOM_FOREST:
            return None

        config_path = Path(__file__).resolve().parent / self.config.sweep_config
        if not config_path.exists():
            return None

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        logger.info("Loaded custom sweep config from %s", config_path)
        return config

    def _build_sweep_config(self) -> dict[str, Any]:
        """Build sweep configuration for selected model type."""
        custom = self._load_custom_random_forest_config()
        if custom is not None:
            return custom
        return ModelFactory.build_default_sweep_config(self.config.best_model_type)

    def _track_run_result(
        self,
        run: wandb.sdk.wandb_run.Run,
        hyperparameters: dict[str, Any],
        metrics: dict[str, float],
    ) -> None:
        """Store run summary to select best run without external API dependency."""
        self.run_summaries.append(
            SweepRunSummary(
                run_id=run.id,
                run_name=run.name,
                entity=run.entity,
                hyperparameters=hyperparameters,
                metrics=metrics,
            )
        )

    def _train_single_run(self) -> None:
        """Training callback executed by W&B agent for each hyperparameter set."""
        run = wandb.init()
        if run is None:
            raise RuntimeError("W&B run initialization failed")

        try:
            raw_params = dict(wandb.config)
            hyperparameters = ModelFactory.sanitize_params(
                model_type=self.config.best_model_type,
                raw_params=raw_params,
                random_state=self.config.random_state,
            )

            model = train_model(
                x_train=self.x_train,
                y_train=self.y_train,
                model_type=self.config.best_model_type,
                params=hyperparameters,
            )

            metrics = evaluate_model(model=model, x_test=self.x_test, y_test=self.y_test)
            feature_importances = extract_feature_importances(model, self.feature_names)

            wandb.log(
                {
                    **hyperparameters,
                    **metrics,
                    **{
                        f"feature_importance_{name}": value
                        for name, value in list(feature_importances.items())[:10]
                    },
                }
            )

            self._track_run_result(run=run, hyperparameters=hyperparameters, metrics=metrics)

            logger.info(
                "Run %s completed | %s=%.4f",
                run.name,
                "wmape",
                metrics.get("wmape", float("nan")),
            )

        except Exception as error:
            logger.exception("Sweep run failed: %s", error)
            wandb.log({"error": str(error), "mape": 999.9, "wmape": 999.9})
            raise
        finally:
            run.finish()

    @staticmethod
    def _metric_sort_value(summary: SweepRunSummary, metric_name: str, goal: str) -> float:
        """Return a sortable value even when metric is missing."""
        value = summary.metrics.get(metric_name)
        if value is not None:
            return value

        if goal == "maximize":
            return float("-inf")
        return float("inf")

    def _select_best_run(self, metric_name: str, goal: str) -> SweepRunSummary:
        """Select best run from in-memory run summaries."""
        if not self.run_summaries:
            raise RuntimeError("No successful sweep runs were recorded")

        if goal == "maximize":
            return max(
                self.run_summaries,
                key=lambda summary: self._metric_sort_value(summary, metric_name, goal),
            )

        return min(
            self.run_summaries,
            key=lambda summary: self._metric_sort_value(summary, metric_name, goal),
        )

    def _build_sweep_url(self, sweep_id: str, entity: str | None) -> str:
        """Build human-readable sweep dashboard URL."""
        owner = entity or os.getenv("WANDB_ENTITY")
        if owner:
            return f"https://wandb.ai/{owner}/{self.config.wandb_project}/sweeps/{sweep_id}"
        return f"https://wandb.ai/{self.config.wandb_project}/sweeps/{sweep_id}"

    def _persist_best_params(self, sweep_id: str, best_run: SweepRunSummary) -> Path:
        """Validate and persist best sweep parameters to YAML output."""
        required_keys = ModelFactory.required_param_keys(self.config.best_model_type)
        missing = sorted(required_keys - set(best_run.hyperparameters.keys()))
        if missing:
            raise ValueError(
                "Missing required hyperparameters in best run: "
                f"{missing} for model {self.config.best_model_type.value}"
            )

        result = SweepBestParams(
            sweep_id=sweep_id,
            best_run_id=best_run.run_id,
            best_run_name=best_run.run_name,
            model_type=self.config.best_model_type,
            hyperparameters=best_run.hyperparameters,
            metrics=best_run.metrics,
            sweep_url=self._build_sweep_url(sweep_id=sweep_id, entity=best_run.entity),
        )

        output_path = Path(__file__).resolve().parent / "best_params.yaml"
        with output_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(result.model_dump(mode="json"), file, sort_keys=False)

        logger.info("Best sweep parameters saved to %s", output_path)
        return output_path

    def run(self) -> tuple[str, Path]:
        """Execute the complete Bayesian sweep workflow."""
        logger.info("=" * 70)
        logger.info("STEP 6: HYPERPARAMETER SWEEP")
        logger.info("=" * 70)
        logger.info("Model selected for tuning: %s", self.config.best_model_type.value)
        logger.info("Sweep runs: %s", self.config.sweep_count)

        self._load_data_once()
        sweep_config = self._build_sweep_config()

        metric_name = sweep_config.get("metric", {}).get("name", "wmape")
        goal = sweep_config.get("metric", {}).get("goal", "minimize")

        sweep_id = wandb.sweep(sweep=sweep_config, project=self.config.wandb_project)
        logger.info("W&B sweep created: %s", sweep_id)

        wandb.agent(
            sweep_id,
            function=self._train_single_run,
            count=self.config.sweep_count,
            project=self.config.wandb_project,
        )

        best_run = self._select_best_run(metric_name=metric_name, goal=goal)
        output_path = self._persist_best_params(sweep_id=sweep_id, best_run=best_run)

        logger.info("Best run: %s (%s)", best_run.run_name, best_run.run_id)
        logger.info("Primary metric (%s): %.4f", metric_name, best_run.metrics[metric_name])

        return sweep_id, output_path


def parse_args() -> SweepExecutionConfig:
    """Parse and validate CLI inputs."""
    parser = argparse.ArgumentParser(description="W&B Bayesian Sweep for model fine-tuning")

    parser.add_argument("--train_artifact_name", type=str, required=True)
    parser.add_argument("--test_artifact_name", type=str, required=True)
    parser.add_argument("--gcs_train_path", type=str, required=True)
    parser.add_argument("--gcs_test_path", type=str, required=True)
    parser.add_argument("--bucket_name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--best_model_type", type=str, default="RandomForest")
    parser.add_argument("--target_column", type=str, default="median_house_value")
    parser.add_argument("--sweep_count", type=int, default=5)
    parser.add_argument("--sweep_config", type=str, default="sweep_config.yaml")
    parser.add_argument("--random_state", type=int, default=42)

    arguments = parser.parse_args()
    return SweepExecutionConfig(**vars(arguments))


def main() -> None:
    """CLI entry point for sweep step."""
    config = parse_args()
    workflow = BayesianSweepWorkflow(config=config)
    workflow.run()


if __name__ == "__main__":
    main()
