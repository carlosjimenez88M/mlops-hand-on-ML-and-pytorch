"""
Step #2 - Preprocessing and Imputation
Description: Load data from GCS, preprocess, and save back to GCS
Author: Carlos Daniel Jiménez
Date: 2025-11-28
"""

#=======================#
# ----- Libraries ----- #
#=======================#

import argparse
import logging
import sys
from datetime import datetime

import wandb
import matplotlib.pyplot as plt
from pydantic import ValidationError

from models import PreprocessingConfig
from preprocessor import DataPreprocessor
from config import settings

# ================================#
# ---- Logger Configuration ---- #
# ================================#
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def main(args: argparse.Namespace) -> int:
    """
    Main function.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    logger.info("=" * 70)
    logger.info("STEP 2: DATA PREPROCESSING AND IMPUTATION")
    logger.info("=" * 70)

    try:
        # 1. Validate and create configuration
        config = PreprocessingConfig(
            input_artifact_name=args.input_artifact_name,
            gcs_input_path=args.gcs_input_path,
            gcs_output_path=args.gcs_output_path,
            artifact_name=args.artifact_name,
            artifact_type=args.artifact_type,
            artifact_description=args.artifact_description,
            bucket_name=args.bucket_name,
            wandb_project=args.wandb_project,
            imputation_strategy=args.imputation_strategy,
            create_features=args.create_features
        )

        logger.info(f"Configuration validated:")
        logger.info(f"  Input: gs://{config.bucket_name}/{config.gcs_input_path}")
        logger.info(f"  Output: gs://{config.bucket_name}/{config.gcs_output_path}")
        logger.info(f"  W&B Project: {config.wandb_project}")
        logger.info(f"  Imputation Strategy: {config.imputation_strategy}")
        logger.info(f"  Create Features: {config.create_features}")

    except ValidationError as e:
        logger.error("Configuration validation error:")
        for error in e.errors():
            logger.error(f"  - {error['loc'][0]}: {error['msg']}")
        return 1

    # 2. Initialize preprocessor
    preprocessor = DataPreprocessor(config)

    # 3. Initialize W&B
    logger.info("\nInitializing Weights & Biases...")

    try:
        with wandb.init(
            project=config.wandb_project,
            job_type="preprocessing",
            name=f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=["preprocessing", "imputation", "feature-engineering", "pipeline-step-2"],
            config=config.model_dump(mode='json')
        ) as run:

            # 4. Execute preprocessing
            result = preprocessor.run()

            if not result.success:
                logger.error(f"Preprocessing failed: {result.error_message}")
                return 1

            # 4.5. Log visualizations to W&B (if auto imputation was used)
            if config.imputation_strategy == "auto" and preprocessor.imputation_analyzer:
                logger.info("\nLogging visualizations to W&B...")
                plots = preprocessor.get_imputation_plots()

                for plot_name, fig in plots.items():
                    logger.info(f"  - Uploading {plot_name}...")
                    wandb.log({plot_name: wandb.Image(fig)})
                    plt.close(fig)  # Close to free memory

                logger.info(" Visualizations uploaded successfully")

            # 5. Create metadata for W&B
            metadata = preprocessor.create_wandb_metadata(result)

            # 6. Create artifact in W&B
            logger.info("\nCreating artifact in W&B...")
            artifact = wandb.Artifact(
                name=config.artifact_name,
                type=config.artifact_type,
                description=config.artifact_description or "Preprocessed dataset",
                metadata=metadata.to_dict()
            )

            # Add reference to GCS
            artifact.add_reference(
                result.gcs_output_uri,
                name="gcs_location",
                checksum=False
            )

            # 7. Log artifact
            logger.info("Logging artifact to W&B...")
            run.log_artifact(artifact)
            artifact.wait()

            # 8. Log metrics
            run.summary.update({
                'input_rows': result.stats.input_rows,
                'output_rows': result.stats.output_rows,
                'rows_dropped': result.stats.rows_dropped,
                'input_columns': result.stats.input_columns,
                'output_columns': result.stats.output_columns,
                'columns_added': result.stats.columns_added,
                'new_features_count': len(result.stats.new_features),
                'input_size_mb': result.stats.input_size_mb,
                'output_size_mb': result.stats.output_size_mb,
                'imputation_strategy': config.imputation_strategy,
                'gcs_output_uri': result.gcs_output_uri
            })

            # Log new features as list
            if result.stats.new_features:
                run.summary['new_features'] = result.stats.new_features

            # 9. Final report
            logger.info("\n" + "=" * 70)
            logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"  - Input GCS URI: {result.gcs_input_uri}")
            logger.info(f"  - Output GCS URI: {result.gcs_output_uri}")
            logger.info(f"  - W&B Artifact: {config.artifact_name}")
            logger.info(f"  - Input: {result.stats.input_rows:,} rows, {result.stats.input_columns} cols")
            logger.info(f"  - Output: {result.stats.output_rows:,} rows, {result.stats.output_columns} cols")
            logger.info(f"  - Rows dropped: {result.stats.rows_dropped:,}")
            logger.info(f"  - New features: {len(result.stats.new_features)}")
            if result.stats.new_features:
                for feature in result.stats.new_features:
                    logger.info(f"      • {feature}")
            logger.info("=" * 70)

            return 0

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and impute data from GCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage example:
  python main.py \\
    --input_artifact_name "housing_data_raw" \\
    --gcs_input_path "data/01-raw/housing.csv" \\
    --gcs_output_path "data/02-processed/housing_processed.csv" \\
    --artifact_name "housing_data_processed" \\
    --bucket_name "my-bucket"

This component:
  1. Downloads data from GCS
  2. Handles missing values (imputation)
  3. Creates engineered features
  4. Uploads processed data back to GCS
  5. Logs artifact to W&B
        """
    )

    parser.add_argument(
        "--input_artifact_name",
        type=str,
        required=True,
        help="Name of the input artifact from W&B (step 1)"
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
        default="data/02-processed/housing_processed.csv",
        help="GCS path for output data (without gs://bucket/)"
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        default="housing_data_processed",
        help="Name of the output artifact in W&B"
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        default="processed_data",
        help="Type of artifact (processed_data, clean_data, etc.)"
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        default="",
        help="Description of the artifact"
    )

    parser.add_argument(
        "--bucket_name",
        type=str,
        default=settings.GCS_BUCKET_NAME,
        help="GCS bucket name (without gs://)"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=settings.WANDB_PROJECT,
        help="Project name in W&B"
    )

    parser.add_argument(
        "--imputation_strategy",
        type=str,
        choices=["mean", "median", "mode", "drop", "auto"],
        default="auto",
        help="Strategy for handling missing values. 'auto' compares all methods and selects the best one based on RMSE."
    )

    parser.add_argument(
        "--create_features",
        type=bool,
        default=True,
        help="Whether to create engineered features"
    )

    args = parser.parse_args()

    sys.exit(main(args))
