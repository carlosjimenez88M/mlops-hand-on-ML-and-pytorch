"""
Data Segregation module for housing data
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

from __future__ import annotations
import io
import sys
import logging
import pandas as pd
from typing import Optional, Tuple, Dict
from pathlib import Path
from sklearn.model_selection import train_test_split
from google.cloud import storage
import wandb

from config import settings
from models import SegregationConfig, SegregationResult

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


class DataSegregator:
    """
    Data segregation component for housing data.
    Splits data into train and test sets and uploads to GCS.
    """

    def __init__(self, config: SegregationConfig):
        """Initialize data segregator with configuration."""
        self.config = config
        self.storage_client: Optional[storage.Client] = None
        self.bucket: Optional[storage.Bucket] = None

        self._init_gcs_client()

    def _init_gcs_client(self) -> None:
        """Initialize GCS client."""
        try:
            import os
            if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') == '':
                os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)

            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.config.bucket_name)

            if not self.bucket.exists():
                raise ValueError(f"Bucket {self.config.bucket_name} does not exist")

            logger.info(f"Connected to GCS: gs://{self.config.bucket_name}")

        except Exception as e:
            logger.error(f"Error connecting to GCS: {e}")
            raise RuntimeError(
                "GCS is required for this component. "
                "Check your configuration and credentials."
            ) from e

    def download_from_gcs(self) -> pd.DataFrame:
        """Download data from GCS."""
        try:
            blob_path = self.config.gcs_input_path
            logger.info(f"Downloading from GCS: gs://{self.config.bucket_name}/{blob_path}")

            blob = self.bucket.blob(blob_path)
            content = blob.download_as_bytes()

            df = pd.read_csv(io.BytesIO(content))
            logger.info(f"Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

            return df

        except Exception as e:
            logger.error(f"Error downloading from GCS: {e}")
            raise

    def upload_to_gcs(self, df: pd.DataFrame, gcs_path: str) -> str:
        """Upload data to GCS."""
        try:
            logger.info(f"Uploading to GCS: gs://{self.config.bucket_name}/{gcs_path}")

            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            blob = self.bucket.blob(gcs_path)
            blob.upload_from_file(csv_buffer, content_type='text/csv')

            gcs_uri = f"gs://{self.config.bucket_name}/{gcs_path}"
            logger.info(f"Uploaded to: {gcs_uri}")

            return gcs_uri

        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")
            raise

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        try:
            logger.info("Splitting data into train and test sets...")

            if self.config.target_column not in df.columns:
                raise KeyError(f"Target column '{self.config.target_column}' not found in dataset")

            logger.info(f"Separating target column: {self.config.target_column}")
            X = df.drop(columns=[self.config.target_column])
            y = df[self.config.target_column]

            logger.info(f"Performing train/test split (test_size={self.config.test_size})")
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

            train_data = X_train.copy()
            train_data[self.config.target_column] = y_train

            test_data = X_test.copy()
            test_data[self.config.target_column] = y_test

            logger.info(f"Train set: {train_data.shape[0]} samples")
            logger.info(f"Test set: {test_data.shape[0]} samples")

            return train_data, test_data

        except Exception as e:
            logger.error(f"Error during data split: {e}")
            raise

    def run(self) -> Dict[str, any]:
        """Execute complete data segregation workflow."""
        try:
            logger.info("=" * 70)
            logger.info("DATA SEGREGATION WORKFLOW")
            logger.info("=" * 70)

            df = self.download_from_gcs()

            train_data, test_data = self.split_data(df)

            train_gcs_uri = self.upload_to_gcs(train_data, self.config.gcs_train_output_path)
            test_gcs_uri = self.upload_to_gcs(test_data, self.config.gcs_test_output_path)

            result = SegregationResult(
                input_shape=(df.shape[0], df.shape[1]),
                train_shape=(train_data.shape[0], train_data.shape[1]),
                test_shape=(test_data.shape[0], test_data.shape[1]),
                target_column=self.config.target_column,
                test_size=self.config.test_size,
                train_gcs_uri=train_gcs_uri,
                test_gcs_uri=test_gcs_uri,
                train_samples=train_data.shape[0],
                test_samples=test_data.shape[0]
            )

            logger.info("=" * 70)
            logger.info("DATA SEGREGATION COMPLETED")
            logger.info("=" * 70)
            logger.info(f"Input: {result.input_shape}")
            logger.info(f"Train: {result.train_shape} ({result.train_samples} samples)")
            logger.info(f"Test: {result.test_shape} ({result.test_samples} samples)")
            logger.info(f"Train GCS URI: {result.train_gcs_uri}")
            logger.info(f"Test GCS URI: {result.test_gcs_uri}")
            logger.info("=" * 70)

            return {
                "result": result,
                "train_data": train_data,
                "test_data": test_data
            }

        except Exception as e:
            logger.error(f"Data segregation failed: {e}")
            raise
