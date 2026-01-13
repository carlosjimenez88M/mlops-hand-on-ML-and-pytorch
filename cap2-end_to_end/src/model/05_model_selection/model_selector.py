"""
Model selection module for housing price prediction
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

from __future__ import annotations
import io
import sys
import logging
import pandas as pd
import numpy as np
import pickle
from typing import Optional, Dict, Any
from pathlib import Path
from google.cloud import storage
import wandb

from config import settings
from models import ModelSelectionConfig, ModelSelectionResult, ModelResult, ModelMetrics
from utils import (
    get_available_models,
    get_default_param_grids,
    train_model_with_gridsearch,
    evaluate_model
)

try:
    sys.path.insert(0, str(__file__).rsplit('/', 6)[0])
    from src.utils.colored_logger import setup_colored_logger
    logger = setup_colored_logger(__name__)
except (ImportError, Exception):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Model selection component for housing data.
    Trains multiple models and selects the best one.
    """

    def __init__(self, config: ModelSelectionConfig):
        """Initialize model selector with configuration."""
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

    def download_from_gcs(self, gcs_path: str) -> pd.DataFrame:
        """Download data from GCS."""
        try:
            logger.info(f"Downloading from GCS: gs://{self.config.bucket_name}/{gcs_path}")

            blob = self.bucket.blob(gcs_path)
            content = blob.download_as_bytes()

            df = pd.read_csv(io.BytesIO(content))
            logger.info(f"Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

            return df

        except Exception as e:
            logger.error(f"Error downloading from GCS: {e}")
            raise

    def upload_model_to_gcs(self, model: Any, gcs_path: str) -> str:
        """Upload trained model to GCS."""
        try:
            logger.info(f"Uploading model to GCS: gs://{self.config.bucket_name}/{gcs_path}")

            model_buffer = io.BytesIO()
            pickle.dump(model, model_buffer)
            model_buffer.seek(0)

            blob = self.bucket.blob(gcs_path)
            blob.upload_from_file(model_buffer, content_type='application/octet-stream')

            gcs_uri = f"gs://{self.config.bucket_name}/{gcs_path}"
            logger.info(f"Uploaded model to: {gcs_uri}")

            return gcs_uri

        except Exception as e:
            logger.error(f"Error uploading model to GCS: {e}")
            raise

    def train_and_evaluate_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate all models.

        Returns:
            Dict with results for each model
        """
        models = get_available_models()
        param_grids = get_default_param_grids()

        results = {}

        logger.info("=" * 70)
        logger.info(f"TRAINING {len(models)} MODELS")
        logger.info("=" * 70)

        for i, (model_name, model) in enumerate(models.items(), 1):
            logger.info(f"\n[{i}/{len(models)}] Training {model_name}...")

            # Train model with grid search
            best_model, best_params, training_time = train_model_with_gridsearch(
                model,
                param_grids[model_name],
                X_train,
                y_train
            )

            # Evaluate model
            metrics = evaluate_model(best_model, X_test, y_test)

            # Store results
            results[model_name] = {
                "model": best_model,
                "best_params": best_params,
                "metrics": metrics,
                "training_time": training_time
            }

            logger.info(f"  MAPE: {metrics['mape']:.2f}% | R²: {metrics['r2']:.4f} | "
                       f"Within-10%: {metrics['within_10pct']:.1f}% | "
                       f"Time: {training_time:.2f}s")

        return results

    def get_best_model(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Get best model based on MAPE metric (lower is better).

        MAPE (Mean Absolute Percentage Error) is the primary business metric
        as it's easily interpretable: 5% MAPE means predictions are off by 5% on average.

        Args:
            results: Dict with all model results

        Returns:
            Name of best model
        """
        best_model_name = min(
            results.keys(),
            key=lambda k: results[k]["metrics"]["mape"]
        )
        return best_model_name

    def run(self) -> Dict[str, Any]:
        """Execute complete model selection workflow."""
        try:
            logger.info("=" * 70)
            logger.info("MODEL SELECTION WORKFLOW")
            logger.info("=" * 70)

            # Download train and test data
            train_df = self.download_from_gcs(self.config.gcs_train_path)
            test_df = self.download_from_gcs(self.config.gcs_test_path)

            # Prepare data
            X_train = train_df.drop(columns=[self.config.target_column])
            y_train = train_df[self.config.target_column]
            X_test = test_df.drop(columns=[self.config.target_column])
            y_test = test_df[self.config.target_column]

            logger.info(f"\nData prepared:")
            logger.info(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            logger.info(f"  Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            logger.info(f"  Target: {self.config.target_column}\n")

            # Train and evaluate all models
            results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)

            # Get best model
            best_model_name = self.get_best_model(results)
            best_model = results[best_model_name]["model"]
            best_metrics = results[best_model_name]["metrics"]

            logger.info("\n" + "=" * 70)
            logger.info(f"🏆 BEST MODEL: {best_model_name}")
            logger.info("=" * 70)
            logger.info("Business Metrics:")
            logger.info(f"  MAPE (Mean APE): {best_metrics['mape']:.2f}%")
            logger.info(f"  Median APE: {best_metrics['median_ape']:.2f}%")
            logger.info(f"  Within ±5%: {best_metrics['within_5pct']:.1f}%")
            logger.info(f"  Within ±10%: {best_metrics['within_10pct']:.1f}%")
            logger.info(f"  Within ±15%: {best_metrics['within_15pct']:.1f}%")
            logger.info("\nTraditional Metrics:")
            logger.info(f"  R²: {best_metrics['r2']:.4f}")
            logger.info(f"  RMSE: ${best_metrics['rmse']:,.2f}")
            logger.info(f"  MAE: ${best_metrics['mae']:,.2f}")
            logger.info("=" * 70)

            # Upload best model to GCS
            model_gcs_path = f"models/05-selection/{best_model_name.lower()}_best.pkl"
            model_gcs_uri = self.upload_model_to_gcs(best_model, model_gcs_path)

            # Create result object
            result = ModelSelectionResult(
                best_model_name=best_model_name,
                all_results={
                    name: {
                        "metrics": res["metrics"],
                        "best_params": res["best_params"],
                        "training_time": res["training_time"]
                    }
                    for name, res in results.items()
                },
                total_models_trained=len(results),
                train_samples=X_train.shape[0],
                test_samples=X_test.shape[0],
                num_features=X_train.shape[1],
                target_column=self.config.target_column
            )

            logger.info("\n" + "=" * 70)
            logger.info("MODEL SELECTION COMPLETED")
            logger.info("=" * 70)

            return {
                "result": result,
                "best_model": best_model,
                "best_model_name": best_model_name,
                "model_gcs_uri": model_gcs_uri,
                "all_results": results
            }

        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            raise
