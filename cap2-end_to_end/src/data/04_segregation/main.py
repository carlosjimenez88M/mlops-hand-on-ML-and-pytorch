"""
Data Segregation Step - Main Entry Point
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

import argparse
import logging
import sys

from segregator import DataSegregator

import wandb
from models import SegregationConfig

try:
    sys.path.insert(0, str(__file__).rsplit("/", 5)[0])
    from src.utils.colored_logger import setup_colored_logger

    logger = setup_colored_logger(__name__)
except (ImportError, Exception):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)


def main():
    """Main entry point for data segregation step."""
    parser = argparse.ArgumentParser(description="Data Segregation for Housing Data")

    parser.add_argument(
        "--input_artifact_name", type=str, required=True, help="Name of the input artifact from W&B"
    )

    parser.add_argument(
        "--gcs_input_path",
        type=str,
        required=True,
        help="GCS path to input data (without gs://bucket/)",
    )

    parser.add_argument(
        "--gcs_train_output_path",
        type=str,
        required=True,
        help="GCS path for train data (without gs://bucket/)",
    )

    parser.add_argument(
        "--gcs_test_output_path",
        type=str,
        required=True,
        help="GCS path for test data (without gs://bucket/)",
    )

    parser.add_argument(
        "--artifact_root", type=str, required=True, help="Root name for output artifacts"
    )

    parser.add_argument(
        "--artifact_type", type=str, default="segregated_data", help="Type of artifact"
    )

    parser.add_argument("--artifact_description", type=str, default="", help="Artifact description")

    parser.add_argument("--bucket_name", type=str, required=True, help="GCS bucket name")

    parser.add_argument(
        "--wandb_project", type=str, default="housing-mlops-gcp", help="W&B Project"
    )

    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Fraction of data for test set"
    )

    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state for reproducibility"
    )

    parser.add_argument(
        "--target_column", type=str, default="median_house_value", help="Target column name"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("STEP 4: DATA SEGREGATION")
    logger.info("=" * 70)
    logger.info("Configuration validated:")
    logger.info(f"  Input: gs://{args.bucket_name}/{args.gcs_input_path}")
    logger.info(f"  Train Output: gs://{args.bucket_name}/{args.gcs_train_output_path}")
    logger.info(f"  Test Output: gs://{args.bucket_name}/{args.gcs_test_output_path}")
    logger.info(f"  W&B Project: {args.wandb_project}")
    logger.info(f"  Test Size: {args.test_size}")
    logger.info(f"  Target Column: {args.target_column}")

    # Configure W&B with explicit settings for CI/CD
    wandb_settings = wandb.Settings(console="wrap")

    run = wandb.init(
        project=args.wandb_project,
        job_type="data_segregation",
        name=f"segregation_{wandb.util.generate_id()}",
        settings=wandb_settings,
    )

    try:
        config = SegregationConfig(
            input_artifact_name=args.input_artifact_name,
            gcs_input_path=args.gcs_input_path,
            gcs_train_output_path=args.gcs_train_output_path,
            gcs_test_output_path=args.gcs_test_output_path,
            artifact_root=args.artifact_root,
            artifact_type=args.artifact_type,
            artifact_description=args.artifact_description,
            bucket_name=args.bucket_name,
            wandb_project=args.wandb_project,
            test_size=args.test_size,
            random_state=args.random_state,
            target_column=args.target_column,
        )

        segregator = DataSegregator(config)

        result_data = segregator.run()

        logger.info("Logging to W&B...")
        wandb.log(
            {
                "train_samples": result_data["result"].train_samples,
                "test_samples": result_data["result"].test_samples,
                "test_size": args.test_size,
                "train_gcs_uri": result_data["result"].train_gcs_uri,
                "test_gcs_uri": result_data["result"].test_gcs_uri,
            }
        )

        # Create and log distribution plots
        from pathlib import Path

        plot_path = segregator.create_distribution_plots(
            result_data["train_data"], result_data["test_data"]
        )
        if plot_path and plot_path.exists():
            wandb.log({"train_test_distributions": wandb.Image(str(plot_path))})
            logger.info("Distribution plots logged to W&B")

        # Create and log statistics table
        stats_table = segregator.create_statistics_table(
            result_data["train_data"], result_data["test_data"]
        )
        if stats_table:
            wandb.log({"train_test_statistics": stats_table})
            logger.info("Statistics table logged to W&B")

        # Save local CSV files and log as artifacts
        artifacts_dir = Path("artifacts/segregation")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        train_csv_path = artifacts_dir / "train_data.csv"
        test_csv_path = artifacts_dir / "test_data.csv"

        result_data["train_data"].to_csv(train_csv_path, index=False)
        result_data["test_data"].to_csv(test_csv_path, index=False)
        logger.info(f"Local CSV files saved to: {artifacts_dir}")

        # Log train artifact (with actual CSV file)
        train_artifact = wandb.Artifact(
            name=f"{args.artifact_root}_train",
            type=args.artifact_type,
            description=args.artifact_description or "Train split of housing data",
        )
        train_artifact.add_file(str(train_csv_path), name="train_data.csv")
        train_artifact.add_reference(
            result_data["result"].train_gcs_uri, name="train_data_gcs_reference"
        )
        run.log_artifact(train_artifact)
        logger.info("Train artifact logged to W&B")

        # Log test artifact (with actual CSV file)
        test_artifact = wandb.Artifact(
            name=f"{args.artifact_root}_test",
            type=args.artifact_type,
            description=args.artifact_description or "Test split of housing data",
        )
        test_artifact.add_file(str(test_csv_path), name="test_data.csv")
        test_artifact.add_reference(
            result_data["result"].test_gcs_uri, name="test_data_gcs_reference"
        )
        run.log_artifact(test_artifact)
        logger.info("Test artifact logged to W&B")

        logger.info("Data segregation completed successfully!")

    except Exception as e:
        logger.error(f"Data segregation failed: {e}")
        run.finish(exit_code=1)
        raise

    run.finish()


if __name__ == "__main__":
    main()
