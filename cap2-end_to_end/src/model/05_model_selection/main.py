"""
Model Selection Step - Main Entry Point
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml
from model_selector import ModelSelector

import wandb
from models import (
    BestModelSummary,
    ModelCvMetrics,
    ModelMetrics,
    ModelSelectionArtifact,
    ModelSelectionConfig,
)

try:
    sys.path.insert(0, str(__file__).rsplit("/", 6)[0])
    from src.utils.colored_logger import setup_colored_logger

    logger = setup_colored_logger(__name__)
except (ImportError, Exception):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)


def save_model_selection_summary(result_data: dict) -> Path:
    """Persist best-model summary for downstream sweep step."""
    best_result = result_data["all_results"][result_data["best_model_name"]]
    summary = ModelSelectionArtifact(
        best_model=BestModelSummary(
            best_model_name=result_data["best_model_name"],
            metrics=ModelMetrics(**best_result["metrics"]),
            cv_metrics=ModelCvMetrics(**best_result["cv_metrics"]),
            best_params=best_result["best_params"],
            training_time=best_result["training_time"],
        ),
        all_model_names=sorted(result_data["all_results"].keys()),
    )

    summary_path = Path(__file__).resolve().parent / "best_model_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(summary.model_dump(mode="json"), file, sort_keys=False)

    return summary_path


def main() -> None:
    """Main entry point for model selection step."""
    parser = ArgumentParser(description="Model Selection for Housing Price Prediction")

    parser.add_argument(
        "--train_artifact_name",
        type=str,
        required=True,
        help="Name of the training data artifact from W&B",
    )

    parser.add_argument(
        "--test_artifact_name",
        type=str,
        required=True,
        help="Name of the test data artifact from W&B",
    )

    parser.add_argument(
        "--gcs_train_path",
        type=str,
        required=True,
        help="GCS path to training data (without gs://bucket/)",
    )

    parser.add_argument(
        "--gcs_test_path",
        type=str,
        required=True,
        help="GCS path to test data (without gs://bucket/)",
    )

    parser.add_argument("--bucket_name", type=str, required=True, help="GCS bucket name")

    parser.add_argument(
        "--wandb_project", type=str, default="housing-mlops-gcp", help="W&B Project"
    )

    parser.add_argument(
        "--target_column", type=str, default="median_house_value", help="Target column name"
    )

    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state for reproducibility"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("STEP 5: MODEL SELECTION")
    logger.info("=" * 70)
    logger.info("Configuration validated:")
    logger.info(f"  Train: gs://{args.bucket_name}/{args.gcs_train_path}")
    logger.info(f"  Test: gs://{args.bucket_name}/{args.gcs_test_path}")
    logger.info(f"  W&B Project: {args.wandb_project}")
    logger.info(f"  Target Column: {args.target_column}")

    run = wandb.init(
        project=args.wandb_project,
        job_type="model_selection",
        name=f"model_selection_{wandb.util.generate_id()}",
    )

    try:
        config = ModelSelectionConfig(
            train_artifact_name=args.train_artifact_name,
            test_artifact_name=args.test_artifact_name,
            gcs_train_path=args.gcs_train_path,
            gcs_test_path=args.gcs_test_path,
            bucket_name=args.bucket_name,
            wandb_project=args.wandb_project,
            target_column=args.target_column,
            random_state=args.random_state,
        )

        selector = ModelSelector(config)

        result_data = selector.run()
        summary_path = save_model_selection_summary(result_data)

        logger.info("\nLogging to W&B...")

        # Log best model metrics
        best_result = result_data["all_results"][result_data["best_model_name"]]
        wandb.log(
            {
                "best_model": result_data["best_model_name"],
                # Business metrics
                "best_mape": best_result["metrics"]["mape"],
                "best_median_ape": best_result["metrics"]["median_ape"],
                "best_within_5pct": best_result["metrics"]["within_5pct"],
                "best_within_10pct": best_result["metrics"]["within_10pct"],
                "best_within_15pct": best_result["metrics"]["within_15pct"],
                # Traditional metrics
                "best_r2": best_result["metrics"]["r2"],
                "best_rmse": best_result["metrics"]["rmse"],
                "best_mae": best_result["metrics"]["mae"],
                # Metadata
                "total_models_trained": result_data["result"].total_models_trained,
                "train_samples": result_data["result"].train_samples,
                "test_samples": result_data["result"].test_samples,
                "num_features": result_data["result"].num_features,
                "model_gcs_uri": result_data["model_gcs_uri"],
                "best_model_summary_path": str(summary_path),
            }
        )

        # Log all model results as a table
        model_results_data = []
        for model_name, res in result_data["all_results"].items():
            model_results_data.append(
                [
                    model_name,
                    res["metrics"]["mape"],
                    res["metrics"]["within_10pct"],
                    res["metrics"]["r2"],
                    res["metrics"]["rmse"],
                    res["metrics"]["mae"],
                    res["training_time"],
                ]
            )

        model_table = wandb.Table(
            columns=["Model", "MAPE (%)", "Within-10% (%)", "R²", "RMSE", "MAE", "Time (s)"],
            data=model_results_data,
        )
        wandb.log({"model_comparison": model_table})

        # Create artifact for best model reference
        artifact = wandb.Artifact(
            name="best_model_selection",
            type="model",
            description=f"Best model selected: {result_data['best_model_name']}",
        )
        artifact.add_reference(result_data["model_gcs_uri"], name="best_model.pkl")
        artifact.add_file(str(summary_path), name="best_model_summary.yaml")
        run.log_artifact(artifact)

        logger.info("Model selection completed successfully!")

    except Exception as e:
        logger.error(f"Model selection failed: {e}")
        run.finish(exit_code=1)
        raise

    run.finish()


if __name__ == "__main__":
    main()
