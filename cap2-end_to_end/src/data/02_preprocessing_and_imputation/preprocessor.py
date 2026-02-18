"""
Data Preprocessor class - Downloads from GCS, processes, and uploads back
Author: Carlos Daniel Jiménez
Date: 2025-11-28
"""

import io
import logging

# Logger configuration
import sys
from datetime import datetime
from typing import Dict, Optional

import mlflow
import pandas as pd
from google.api_core import exceptions as gcp_exceptions
from google.cloud import storage
from imputation_analyzer import ImputationAnalyzer
from matplotlib.figure import Figure

from models import (
    PreprocessingConfig,
    PreprocessingResult,
    PreprocessingStats,
    WandBArtifactMetadata,
)

try:
    sys.path.insert(0, str(__file__).rsplit("/", 5)[0])
    from src.utils.colored_logger import setup_colored_logger

    logger = setup_colored_logger(__name__)
except (ImportError, Exception):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing operations with GCS."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.storage_client: Optional[storage.Client] = None
        self.bucket: Optional[storage.Bucket] = None
        self.df_input: Optional[pd.DataFrame] = None
        self.df_output: Optional[pd.DataFrame] = None
        self.imputation_analyzer: Optional[ImputationAnalyzer] = None
        self.imputation_metrics: Dict[str, float] = {}
        self.best_imputation_method: str = ""

        self._init_gcs_client()

    def _init_gcs_client(self) -> None:
        """Initializes GCS client."""
        try:
            # If GOOGLE_APPLICATION_CREDENTIALS is empty, unset it to use ADC
            import os

            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") == "":
                os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.config.bucket_name)

            if not self.bucket.exists():
                raise ValueError(f"Bucket {self.config.bucket_name} does not exist")

            logger.info(f"Connected to GCS: gs://{self.config.bucket_name}")

        except Exception as e:
            logger.error(f"Error connecting to GCS: {e}")
            raise RuntimeError(
                "GCS is required for this component. Check your configuration and credentials."
            ) from e

    def download_from_gcs(self) -> pd.DataFrame:
        """
        Downloads data from GCS and loads into DataFrame.

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            RuntimeError: If download fails
        """
        gcs_path = self.config.gcs_input_path
        logger.info(f"Downloading from GCS: gs://{self.config.bucket_name}/{gcs_path}")

        try:
            blob = self.bucket.blob(gcs_path)

            if not blob.exists():
                raise FileNotFoundError(
                    f"File not found in GCS: gs://{self.config.bucket_name}/{gcs_path}"
                )

            # Download to bytes
            content_bytes = blob.download_as_bytes()
            size_mb = len(content_bytes) / (1024 * 1024)

            logger.info(f"Downloaded: {size_mb:.2f} MB")

            # Load into DataFrame (supports CSV and Parquet)
            if gcs_path.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(content_bytes))
            else:
                df = pd.read_csv(io.BytesIO(content_bytes))
            logger.info(f"Loaded DataFrame: {len(df):,} rows, {len(df.columns)} columns")

            self.df_input = df
            return df

        except gcp_exceptions.GoogleAPIError as e:
            logger.error(f"Error downloading from GCS: {e}")
            raise RuntimeError(f"Error downloading from GCS: {e}") from e

    def get_input_stats(self, df: pd.DataFrame) -> dict:
        """Get statistics from input data."""
        return {
            "input_rows": len(df),
            "input_columns": len(df.columns),
            "missing_values_before": df.isnull().sum().to_dict(),
            "input_size_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values according to the configured strategy.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with handled missing values
        """
        logger.info(f"Handling missing values using strategy: {self.config.imputation_strategy}")

        df_processed = df.copy()

        if self.config.imputation_strategy == "auto":
            # Use automatic imputation method selection
            # This compares SimpleImputer (median, mean), KNNImputer, and IterativeImputer (RF)
            df_processed = self.analyze_and_select_best_imputation(df_processed)
            logger.info(
                f"Auto imputation completed using best method: {self.best_imputation_method}"
            )

        elif self.config.imputation_strategy == "drop":
            df_processed = df_processed.dropna()
            logger.info(
                f"Dropped rows with missing values: {len(df) - len(df_processed)} rows removed"
            )

        elif self.config.imputation_strategy == "median":
            # Impute numerical columns with median
            numerical_cols = df_processed.select_dtypes(include=["int64", "float64"]).columns
            for col in numerical_cols:
                if df_processed[col].isnull().any():
                    median_value = df_processed[col].median()
                    df_processed[col].fillna(median_value, inplace=True)
                    logger.info(f"Imputed '{col}' with median: {median_value:.2f}")

        elif self.config.imputation_strategy == "mean":
            # Impute numerical columns with mean
            numerical_cols = df_processed.select_dtypes(include=["int64", "float64"]).columns
            for col in numerical_cols:
                if df_processed[col].isnull().any():
                    mean_value = df_processed[col].mean()
                    df_processed[col].fillna(mean_value, inplace=True)
                    logger.info(f"Imputed '{col}' with mean: {mean_value:.2f}")

        elif self.config.imputation_strategy == "mode":
            # Impute all columns with mode
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    mode_value = df_processed[col].mode()[0]
                    df_processed[col].fillna(mode_value, inplace=True)
                    logger.info(f"Imputed '{col}' with mode: {mode_value}")

        return df_processed

    def create_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Creates engineered features.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (DataFrame with new features, list of new feature names)
        """
        if not self.config.create_features:
            return df, []

        logger.info("Creating engineered features...")

        df_processed = df.copy()
        new_features = []

        # Feature 1: rooms_per_household
        if "total_rooms" in df.columns and "households" in df.columns:
            df_processed["rooms_per_household"] = (
                df_processed["total_rooms"] / df_processed["households"]
            )
            new_features.append("rooms_per_household")
            logger.info("Created feature: rooms_per_household")

        # Feature 2: bedrooms_per_room
        if "total_bedrooms" in df.columns and "total_rooms" in df.columns:
            df_processed["bedrooms_per_room"] = (
                df_processed["total_bedrooms"] / df_processed["total_rooms"]
            )
            new_features.append("bedrooms_per_room")
            logger.info("Created feature: bedrooms_per_room")

        # Feature 3: population_per_household
        if "population" in df.columns and "households" in df.columns:
            df_processed["population_per_household"] = (
                df_processed["population"] / df_processed["households"]
            )
            new_features.append("population_per_household")
            logger.info("Created feature: population_per_household")

        logger.info(f"Created {len(new_features)} new features")
        return df_processed, new_features

    def upload_to_gcs(self, df: pd.DataFrame) -> str:
        """
        Uploads processed DataFrame to GCS.

        Args:
            df: Processed DataFrame

        Returns:
            str: Full GCS URI

        Raises:
            RuntimeError: If upload fails
        """
        gcs_path = self.config.gcs_output_path

        logger.info(f"Uploading to GCS: gs://{self.config.bucket_name}/{gcs_path}")

        try:
            # Convert DataFrame to bytes (CSV or Parquet)
            if gcs_path.endswith(".parquet"):
                buffer = io.BytesIO()
                df.to_parquet(buffer, index=False, engine="pyarrow")
                buffer.seek(0)
                data_bytes = buffer.getvalue()
            else:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                data_bytes = csv_buffer.getvalue().encode("utf-8")

            # Upload to GCS
            blob = self.bucket.blob(gcs_path)

            # Metadata
            blob.metadata = {
                "uploaded_at": datetime.now().isoformat(),
                "source": "preprocessing_component",
                "component": "02_preprocessing_and_imputation",
                "rows": str(len(df)),
                "columns": str(len(df.columns)),
                "imputation_strategy": self.config.imputation_strategy,
                "file_size_mb": str(len(data_bytes) / (1024 * 1024)),
            }

            blob.upload_from_string(data_bytes, content_type="text/csv")

            gcs_uri = f"gs://{self.config.bucket_name}/{gcs_path}"
            logger.info(f"Uploaded to GCS: {gcs_uri}")

            return gcs_uri

        except gcp_exceptions.GoogleAPIError as e:
            logger.error(f"Error uploading to GCS: {e}")
            raise RuntimeError(f"Error uploading to GCS: {e}") from e

    def run(self) -> PreprocessingResult:
        """
        Executes the complete preprocessing pipeline.

        Returns:
            PreprocessingResult with operation results

        Raises:
            Exception: If any error occurs
        """
        try:
            # 1. Download data from GCS
            df_input = self.download_from_gcs()

            # 2. Get input statistics
            input_stats = self.get_input_stats(df_input)

            # 3. Handle missing values
            df_processed = self.handle_missing_values(df_input)

            # 3.5. Log imputation metrics to MLflow (if auto was used)
            if self.config.imputation_strategy == "auto" and self.imputation_metrics:
                try:
                    logger.info("\nLogging imputation metrics to MLflow...")
                    # Check if there's an active MLflow run
                    active_run = mlflow.active_run()
                    if active_run:
                        for metric_name, metric_value in self.imputation_metrics.items():
                            mlflow.log_metric(metric_name, metric_value)
                        mlflow.log_param("best_imputation_method", self.best_imputation_method)
                        logger.info(
                            f"Logged {len(self.imputation_metrics)} imputation metrics to MLflow"
                        )
                        logger.info(f"Best imputation method: {self.best_imputation_method}")
                    else:
                        logger.warning("No active MLflow run, skipping MLflow logging")
                except Exception as e:
                    logger.warning(f"Could not log to MLflow: {e}")

            # 4. Create engineered features
            df_processed, new_features = self.create_features(df_processed)

            # 5. Get output statistics
            output_stats = {
                "output_rows": len(df_processed),
                "output_columns": len(df_processed.columns),
                "missing_values_after": df_processed.isnull().sum().to_dict(),
                "output_size_mb": df_processed.memory_usage(deep=True).sum() / (1024 * 1024),
                "new_features": new_features,
            }

            # 6. Upload to GCS
            gcs_output_uri = self.upload_to_gcs(df_processed)

            self.df_output = df_processed

            # 7. Create statistics object
            stats = PreprocessingStats(**input_stats, **output_stats, processed_at=datetime.now())

            logger.info("\n" + "=" * 70)
            logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"  - Input rows: {stats.input_rows:,}")
            logger.info(f"  - Output rows: {stats.output_rows:,}")
            logger.info(f"  - Rows dropped: {stats.rows_dropped:,}")
            logger.info(f"  - Input columns: {stats.input_columns}")
            logger.info(f"  - Output columns: {stats.output_columns}")
            logger.info(f"  - New features: {len(new_features)}")
            if new_features:
                logger.info(f"    {', '.join(new_features)}")
            logger.info("=" * 70)

            return PreprocessingResult(
                gcs_input_uri=f"gs://{self.config.bucket_name}/{self.config.gcs_input_path}",
                gcs_output_uri=gcs_output_uri,
                stats=stats,
                artifact_name=self.config.artifact_name,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}", exc_info=True)

            return PreprocessingResult(
                gcs_input_uri=f"gs://{self.config.bucket_name}/{self.config.gcs_input_path}",
                gcs_output_uri=f"gs://{self.config.bucket_name}/{self.config.gcs_output_path}",
                stats=PreprocessingStats(
                    input_rows=0,
                    input_columns=0,
                    output_rows=0,
                    output_columns=0,
                    input_size_mb=0,
                    output_size_mb=0,
                ),
                artifact_name=self.config.artifact_name,
                success=False,
                error_message=str(e),
            )

    def analyze_and_select_best_imputation(
        self, df: pd.DataFrame, target_column: str = "total_bedrooms"
    ) -> pd.DataFrame:
        """
        Analyzes different imputation methods and selects the best one.

        Args:
            df: Input DataFrame
            target_column: Column to analyze

        Returns:
            DataFrame with imputed values using the best method
        """
        logger.info("\n" + "=" * 70)
        logger.info("AUTOMATIC IMPUTATION METHOD SELECTION")
        logger.info("=" * 70)

        # Initialize analyzer
        self.imputation_analyzer = ImputationAnalyzer(df, target_column=target_column)

        # Analyze missing values
        _ = self.imputation_analyzer.analyze_missing_values()

        # Compute and log correlation matrix
        _ = self.imputation_analyzer.compute_correlation_matrix()

        # Compare all methods
        _ = self.imputation_analyzer.compare_all_methods()

        # Get metrics for tracking
        self.imputation_metrics = self.imputation_analyzer.get_metrics_dict()

        # Get best method name with None check
        best_method_key = self.imputation_analyzer.best_method
        if best_method_key and best_method_key in self.imputation_analyzer.results:
            self.best_imputation_method = self.imputation_analyzer.results[
                best_method_key
            ].method_name
        else:
            self.best_imputation_method = "unknown"
            logger.warning("Could not determine best imputation method")

        # Apply best imputer to full dataset
        df_imputed = self.imputation_analyzer.apply_best_imputer(df)

        return df_imputed

    def get_imputation_plots(self) -> Dict[str, Figure]:
        """
        Generates plots for imputation analysis.

        Returns:
            Dictionary of plot names to figures
        """
        if self.imputation_analyzer is None:
            return {}

        plots = {}

        # Correlation heatmap
        corr_matrix = self.imputation_analyzer.compute_correlation_matrix()
        plots["correlation_heatmap"] = self.imputation_analyzer.create_correlation_heatmap(
            corr_matrix
        )

        # Comparison plot
        plots["imputation_comparison"] = self.imputation_analyzer.create_comparison_plot()

        return plots

    def create_wandb_metadata(self, result: PreprocessingResult) -> WandBArtifactMetadata:
        """Creates metadata for W&B artifact."""
        return WandBArtifactMetadata(
            input_artifact=self.config.input_artifact_name,
            input_gcs_uri=result.gcs_input_uri,
            output_gcs_uri=result.gcs_output_uri,
            bucket=self.config.bucket_name,
            input_size_mb=result.stats.input_size_mb,
            output_size_mb=result.stats.output_size_mb,
            input_rows=result.stats.input_rows,
            output_rows=result.stats.output_rows,
            rows_dropped=result.stats.rows_dropped,
            input_columns=result.stats.input_columns,
            output_columns=result.stats.output_columns,
            columns_added=result.stats.columns_added,
            new_features=result.stats.new_features,
            imputation_strategy=self.config.imputation_strategy,
            processed_at=result.stats.processed_at.isoformat(),
        )
