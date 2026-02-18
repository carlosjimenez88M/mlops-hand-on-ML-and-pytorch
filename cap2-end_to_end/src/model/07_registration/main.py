"""Register final tuned model in MLflow and persist serving metadata."""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient

import wandb
from models import (
    RegistrationResult,
    RegistrationRuntimeConfig,
    SupportedModelType,
    SweepBestParamsFile,
)
from utils import (
    GCSDataRepository,
    create_feature_importance_plot,
    evaluate_model,
    prepare_data,
    save_model_locally,
    train_final_model,
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


class RegistrationWorkflow:
    """End-to-end model registration workflow."""

    def __init__(self, config: RegistrationRuntimeConfig):
        self.config = config
        self.repository = GCSDataRepository(bucket_name=config.bucket_name)

    def _load_best_params(self) -> SweepBestParamsFile:
        """Load and validate sweep output file."""
        params_path = Path(self.config.best_params_path)
        if not params_path.exists():
            raise FileNotFoundError(
                f"Best parameters file not found: {params_path}. "
                "Run step 06_sweep before registration."
            )

        with params_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}

        payload.setdefault("model_type", SupportedModelType.RANDOM_FOREST.value)
        return SweepBestParamsFile(**payload)

    @staticmethod
    def _register_model_to_mlflow(
        model: Any,
        model_name: str,
        model_stage: str,
        model_type: SupportedModelType,
        params: dict[str, Any],
        metrics: dict[str, float],
        feature_columns: list[str],
        target_column: str,
        gcs_train_path: str,
        gcs_test_path: str,
    ) -> tuple[str, str, str]:
        """Register model in MLflow Model Registry and enrich version metadata."""
        logger.info("=" * 70)
        logger.info("REGISTERING MODEL TO MLFLOW")
        logger.info("=" * 70)

        client = MlflowClient()
        run = mlflow.active_run()
        if run is None:
            raise RuntimeError("No active MLflow run available for registration")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        mlflow.sklearn.log_model(model, "model")

        try:
            client.create_registered_model(
                name=model_name,
                description=f"Housing price prediction model using {model_type.value}",
            )
            logger.info("Created new registered model: %s", model_name)
        except Exception as error:
            if "already exists" in str(error).lower():
                logger.info("Registered model already exists: %s", model_name)
            else:
                raise

        model_version = client.create_model_version(
            name=model_name, source=model_uri, run_id=run_id
        )
        deployment_alias = model_stage.strip().lower()
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias=deployment_alias,
                version=model_version.version,
            )
            logger.info(
                "Model version %s assigned alias '%s'",
                model_version.version,
                deployment_alias,
            )
        except Exception as alias_error:
            logger.error(
                "Could not set alias '%s' for model '%s' version '%s': %s",
                deployment_alias,
                model_name,
                model_version.version,
                alias_error,
            )
            raise RuntimeError(
                "MLflow alias assignment failed. Configure a compatible MLflow "
                "registry backend or use a version that supports model aliases."
            ) from alias_error

        description = f"""
# Housing Price Prediction Model

**Algorithm:** {model_type.value}

## Hyperparameters
{yaml.safe_dump(params, sort_keys=True)}

## Performance Metrics
{yaml.safe_dump(metrics, sort_keys=True)}

## Features
- Number of features: {len(feature_columns)}
- Target: {target_column}

## Data Sources
- Training: {gcs_train_path}
- Testing: {gcs_test_path}
"""

        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description,
        )

        tags = {
            "algorithm": model_type.value,
            "framework": "sklearn",
            "mae": f"{metrics['mae']:.2f}",
            "rmse": f"{metrics['rmse']:.2f}",
            "r2": f"{metrics['r2']:.4f}",
            "mape": f"{metrics['mape']:.2f}",
            "wmape": f"{metrics['wmape']:.2f}",
            "within_10pct": f"{metrics['within_10pct']:.1f}",
            "n_features": str(len(feature_columns)),
            "target": target_column,
            "deployment_alias": deployment_alias,
        }

        for key, value in tags.items():
            client.set_model_version_tag(model_name, model_version.version, key, value)

        logger.info(
            "Model registration completed for version %s (deployment label: %s)",
            model_version.version,
            model_stage,
        )
        return model_uri, str(model_version.version), run_id

    def run(self) -> RegistrationResult:
        """Execute final training and registration workflow."""
        logger.info("=" * 70)
        logger.info("STEP 7: MODEL REGISTRATION")
        logger.info("=" * 70)

        best_params_data = self._load_best_params()
        model_type = best_params_data.model_type
        params = best_params_data.hyperparameters

        wandb_settings = wandb.Settings(console="wrap")
        wandb_run = wandb.init(
            project=self.config.wandb_project,
            name="model_registration",
            job_type="registration",
            settings=wandb_settings,
        )

        if mlflow.active_run():
            logger.info("Using existing active MLflow run context")
            run_context = nullcontext()
        else:
            logger.info("No active MLflow run found, creating one")
            run_context = mlflow.start_run(run_name="model_registration")

        try:
            with run_context:
                train_df = self.repository.download_dataframe(self.config.gcs_train_path)
                x_train, y_train = prepare_data(train_df, self.config.target_column)

                test_df = self.repository.download_dataframe(self.config.gcs_test_path)
                x_test, y_test = prepare_data(test_df, self.config.target_column)

                feature_columns = x_train.columns.tolist()

                model = train_final_model(
                    x_train=x_train,
                    y_train=y_train,
                    params=params,
                    model_type=model_type,
                )
                metrics = evaluate_model(model=model, x_test=x_test, y_test=y_test)

                mlflow.log_param("model_type", model_type.value)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_param("n_features", len(feature_columns))
                mlflow.log_param("sweep_id", best_params_data.sweep_id)

                wandb.log(
                    {
                        "model_type": model_type.value,
                        **params,
                        **metrics,
                        "n_features": len(feature_columns),
                        "sweep_id": best_params_data.sweep_id,
                    }
                )

                plot_path = create_feature_importance_plot(
                    model=model, feature_names=feature_columns
                )
                if plot_path and plot_path.exists():
                    wandb.log({"feature_importance": wandb.Image(str(plot_path))})
                    mlflow.log_artifact(str(plot_path), artifact_path="plots")

                model_uri, model_version, run_id = self._register_model_to_mlflow(
                    model=model,
                    model_name=self.config.registered_model_name,
                    model_stage=self.config.model_stage,
                    model_type=model_type,
                    params=params,
                    metrics=metrics,
                    feature_columns=feature_columns,
                    target_column=self.config.target_column,
                    gcs_train_path=self.config.gcs_train_path,
                    gcs_test_path=self.config.gcs_test_path,
                )

                model_path = Path("models/trained") / f"{self.config.registered_model_name}.pkl"
                save_model_locally(model=model, output_path=model_path)
                gcs_model_uri = self.repository.upload_pickle(
                    obj=model,
                    gcs_path=f"models/07-registration/{self.config.registered_model_name}.pkl",
                )

                project_root = Path(__file__).resolve().parent.parent.parent.parent
                config_dir = project_root / "configs"
                config_dir.mkdir(parents=True, exist_ok=True)

                model_config = {
                    "model": {
                        "name": self.config.registered_model_name,
                        "version": model_version,
                        "stage": self.config.model_stage,
                        "best_model": model_type.value,
                        "parameters": params,
                        "r2_score": float(metrics["r2"]),
                        "mae": float(metrics["mae"]),
                        "rmse": float(metrics["rmse"]),
                        "mape": float(metrics["mape"]),
                        "target_variable": self.config.target_column,
                        "num_features": len(feature_columns),
                        "feature_columns": feature_columns,
                        "mlflow_run_id": run_id,
                        "mlflow_model_uri": model_uri,
                        "gcs_model_uri": gcs_model_uri,
                        "local_path": str(model_path),
                        "sweep_id": best_params_data.sweep_id,
                    }
                }

                config_path = config_dir / "model_config.yaml"
                with config_path.open("w", encoding="utf-8") as file:
                    yaml.safe_dump(model_config, file, sort_keys=False)

                mlflow.log_artifact(str(config_path), artifact_path="config")

                result = RegistrationResult(
                    model_name=self.config.registered_model_name,
                    model_version=str(model_version),
                    model_stage=self.config.model_stage,
                    model_uri=model_uri,
                    gcs_model_uri=gcs_model_uri,
                    run_id=run_id,
                    model_type=model_type,
                    hyperparameters=params,
                    metrics=metrics,
                    feature_columns=feature_columns,
                )

                wandb.log(
                    {
                        "model_name": result.model_name,
                        "model_version": result.model_version,
                        "model_stage": result.model_stage,
                        "mlflow_run_id": result.run_id,
                        "gcs_model_uri": result.gcs_model_uri,
                    }
                )

                logger.info("Model registration completed successfully")
                return result

        finally:
            if wandb_run is not None:
                wandb_run.finish()


def parse_args() -> RegistrationRuntimeConfig:
    """Parse CLI arguments and validate with pydantic."""
    parser = argparse.ArgumentParser(description="Register final model to MLflow")
    parser.add_argument("--bucket_name", type=str, required=True)
    parser.add_argument("--gcs_train_path", type=str, required=True)
    parser.add_argument("--gcs_test_path", type=str, required=True)
    parser.add_argument("--best_params_path", type=str, required=True)
    parser.add_argument("--registered_model_name", type=str, default="housing_price_model")
    parser.add_argument("--model_stage", type=str, default="Staging")
    parser.add_argument("--target_column", type=str, default="median_house_value")
    parser.add_argument("--wandb_project", type=str, required=True)

    args = parser.parse_args()
    return RegistrationRuntimeConfig(**vars(args))


def main() -> None:
    """CLI entrypoint."""
    config = parse_args()
    workflow = RegistrationWorkflow(config=config)
    workflow.run()


if __name__ == "__main__":
    main()
