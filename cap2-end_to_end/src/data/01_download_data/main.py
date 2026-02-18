"""
Step #1 - Download the data from GCS
Description: Download the raw data directly to GCS bucket (no local storage).
Author: Carlos Daniel Jiménez
Date: 2025-11-25
"""

# =======================#
# ----- Libraries ----- #
# =======================#

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from config import settings
from downloader import DataDownloader
from pydantic import ValidationError

import wandb
from models import DownloadConfig

# ================================#
# ---- Logger Configuration ---- #
# ================================#
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


def main(args: argparse.Namespace) -> int:
    """
    Main function.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    logger.info("=" * 70)
    logger.info("STEP 1: DATA DOWNLOAD (DIRECT TO GCS)")
    logger.info("=" * 70)

    try:
        # 1. Validate and create configuration
        config = DownloadConfig(
            file_url=args.file_url,
            artifact_name=args.artifact_name,
            artifact_type=args.artifact_type,
            artifact_description=args.artifact_description,
            gcs_output_path=args.gcs_output_path,
            bucket_name=args.bucket_name,
            wandb_project=args.wandb_project,
        )

        logger.info("Configuration validated:")
        logger.info(f"  URL: {config.file_url}")
        logger.info(f"  GCS Output: gs://{config.bucket_name}/{config.gcs_output_path}")
        logger.info(f"  W&B Project: {config.wandb_project}")
        logger.info("  Mode: NO local storage")

    except ValidationError as e:
        logger.error("Configuration validation error:")
        for error in e.errors():
            logger.error(f"  - {error['loc'][0]}: {error['msg']}")
        return 1

    # 2. Initialize downloader
    downloader = DataDownloader(config)

    # 3. Initialize W&B
    logger.info("\nInitializing Weights & Biases...")

    try:
        with wandb.init(
            project=config.wandb_project,
            job_type="download_data",
            name=f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=["data-download", "pipeline-step-1", "gcs-only"],
            config=config.model_dump(mode="json"),
        ) as run:
            # 4. Execute download
            result = downloader.run()

            if not result.success:
                logger.error(f"Download failed: {result.error_message}")
                return 1

            # 5. Create metadata for W&B
            metadata = downloader.create_wandb_metadata(result)

            # 6. Create artifact in W&B
            logger.info("\nCreating artifact in W&B...")
            artifact = wandb.Artifact(
                name=config.artifact_name,
                type=config.artifact_type,
                description=config.artifact_description or "Downloaded dataset (GCS only)",
                metadata=metadata.to_dict(),
            )

            artifact.add_reference(result.gcs_uri, name="gcs_location", checksum=False)

            # 7. Log artifact
            logger.info("Logging artifact to W&B...")
            run.log_artifact(artifact)
            artifact.wait()

            # 8. Log metrics
            run.summary.update(
                {
                    "file_size_mb": result.stats.file_size_mb,
                    "gcs_uri": result.gcs_uri,
                    "storage_type": "gcs_only",
                    "has_missing_values": result.stats.has_missing_values,
                }
            )

            if result.stats.n_rows:
                run.summary.update(
                    {"n_rows": result.stats.n_rows, "n_columns": result.stats.n_columns}
                )

            # 9. Final report
            logger.info("\n" + "=" * 70)
            logger.info("DOWNLOAD COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"  - GCS URI: {result.gcs_uri}")
            logger.info(f"  - W&B Artifact: {config.artifact_name}")
            logger.info(f"  - Size: {result.stats.file_size_mb} MB")
            logger.info("  - Storage: GCS ONLY (no local copy)")

            if result.stats.n_rows:
                logger.info(f"  - Rows: {result.stats.n_rows:,}")
                logger.info(f"  - Columns: {result.stats.n_columns}")

                if result.stats.has_missing_values:
                    logger.warning("  There are missing values in the dataset")
                    for col, count in result.stats.missing_values.items():
                        if count > 0:
                            pct = result.stats.missing_percentage(col)
                            logger.warning(f"      - {col}: {count} ({pct:.2f}%)")

            logger.info("=" * 70)

            return 0

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download dataset DIRECTLY to GCS (no local storage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage example:
  python main.py \\
    --file_url "https://example.com/data.tgz" \\
    --artifact_name "my_dataset" \\
    --artifact_type "raw_data" \\
    --gcs_output_path "data/01-raw/housing.csv" \\
    --bucket_name "my-bucket"

IMPORTANT: This component does NOT save files locally.
Everything is stored directly in GCS.
        """,
    )

    parser.add_argument("--file_url", type=str, required=True, help="URL of the file to download")

    parser.add_argument(
        "--artifact_name", type=str, required=True, help="Name of the artifact in W&B"
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        required=True,
        help="Type of artifact (raw_data, processed_data, etc.)",
    )

    parser.add_argument(
        "--artifact_description", type=str, default="", help="Description of the artifact"
    )

    parser.add_argument(
        "--gcs_output_path",
        type=str,
        default="data/01-raw/housing.csv",
        help="Path in GCS (without gs://bucket/)",
    )

    parser.add_argument(
        "--bucket_name",
        type=str,
        default=settings.GCS_BUCKET_NAME,
        help="GCS bucket name (without gs://)",
    )

    parser.add_argument(
        "--wandb_project", type=str, default=settings.WANDB_PROJECT, help="Project name in W&B"
    )

    args = parser.parse_args()

    sys.exit(main(args))
