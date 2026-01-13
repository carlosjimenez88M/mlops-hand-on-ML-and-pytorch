"""
Feature Engineering Step - Main Entry Point
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

import argparse
import sys
import logging
import wandb

from models import FeatureEngineeringConfig
from feature_engineer import FeatureEngineer

try:
    sys.path.insert(0, str(__file__).rsplit('/', 5)[0])
    from src.utils.colored_logger import setup_colored_logger
    logger = setup_colored_logger(__name__)
except (ImportError, Exception):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)


def main():
    """Main entry point for feature engineering step."""
    parser = argparse.ArgumentParser(description="Feature Engineering for Housing Data")

    parser.add_argument(
        "--input_artifact_name",
        type=str,
        required=True,
        help="Name of the input artifact from W&B"
    )

    parser.add_argument(
        "--gcs_input_path",
        type=str,
        required=True,
        help="GCS path to input data (without gs://bucket/)"
    )

    parser.add_argument(
        "--gcs_output_path",
        type=str,
        required=True,
        help="GCS path for processed data (without gs://bucket/)"
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        required=True,
        help="Name of the output artifact in W&B"
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        default="engineered_features",
        help="Type of artifact"
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        default="",
        help="Artifact description"
    )

    parser.add_argument(
        "--bucket_name",
        type=str,
        required=True,
        help="GCS bucket name"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="housing-mlops-gcp",
        help="W&B Project"
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="Number of clusters for geo features"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Gamma parameter for clustering"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("=" * 70)
    logger.info("Configuration validated:")
    logger.info(f"  Input: gs://{args.bucket_name}/{args.gcs_input_path}")
    logger.info(f"  Output: gs://{args.bucket_name}/{args.gcs_output_path}")
    logger.info(f"  W&B Project: {args.wandb_project}")
    logger.info(f"  N Clusters: {args.n_clusters}")
    logger.info(f"  Gamma: {args.gamma}")

    run = wandb.init(
        project=args.wandb_project,
        job_type="feature_engineering",
        name=f"feature_eng_{wandb.util.generate_id()}"
    )

    try:
        config = FeatureEngineeringConfig(
            input_artifact_name=args.input_artifact_name,
            gcs_input_path=args.gcs_input_path,
            gcs_output_path=args.gcs_output_path,
            artifact_name=args.artifact_name,
            artifact_type=args.artifact_type,
            artifact_description=args.artifact_description,
            bucket_name=args.bucket_name,
            wandb_project=args.wandb_project,
            n_clusters=args.n_clusters,
            gamma=args.gamma,
            random_state=args.random_state
        )

        engineer = FeatureEngineer(config)

        result_data = engineer.run()

        logger.info("Logging to W&B...")
        wandb.log({
            "input_shape": result_data["df_shape"],
            "features_added": result_data["result"].features_added,
            "n_clusters": args.n_clusters,
            "gamma": args.gamma,
            "gcs_output_uri": result_data["gcs_uri"]
        })

        artifact = wandb.Artifact(
            name=args.artifact_name,
            type=args.artifact_type,
            description=args.artifact_description or "Feature engineered housing data"
        )

        artifact.add_reference(result_data["gcs_uri"], name="engineered_data.csv")

        run.log_artifact(artifact)

        logger.info("Feature engineering completed successfully!")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        run.finish(exit_code=1)
        raise

    run.finish()


if __name__ == "__main__":
    main()
