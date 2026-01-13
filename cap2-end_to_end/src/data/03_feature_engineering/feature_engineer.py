"""
Feature Engineering module for housing data
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

from __future__ import annotations
import io
import sys
import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from google.cloud import storage
import wandb

from config import settings
from models import FeatureEngineeringConfig, TransformationResult

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


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """Custom transformer for geographical clustering."""

    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        cluster_labels = self.kmeans_.predict(X)
        return np.expand_dims(cluster_labels, axis=1)

    def get_feature_names_out(self, names=None):
        return ["cluster_label"]


class FeatureEngineer:
    """
    Feature engineering component for housing data.
    Applies transformations including:
    - Numerical feature scaling
    - Categorical encoding
    - Geographical clustering
    """

    def __init__(self, config: FeatureEngineeringConfig):
        """Initialize feature engineer with configuration."""
        self.config = config
        self.storage_client: Optional[storage.Client] = None
        self.bucket: Optional[storage.Bucket] = None
        self.preprocessing_pipeline: Optional[ColumnTransformer] = None

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

    def upload_to_gcs(self, df: pd.DataFrame) -> str:
        """Upload processed data to GCS."""
        try:
            blob_path = self.config.gcs_output_path
            logger.info(f"Uploading to GCS: gs://{self.config.bucket_name}/{blob_path}")

            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            blob = self.bucket.blob(blob_path)
            blob.upload_from_file(csv_buffer, content_type='text/csv')

            gcs_uri = f"gs://{self.config.bucket_name}/{blob_path}"
            logger.info(f"Uploaded to: {gcs_uri}")

            return gcs_uri

        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")
            raise

    def create_preprocessing_pipeline(
        self,
        num_attribs: List[str],
        cat_attribs: List[str]
    ) -> ColumnTransformer:
        """Create sklearn preprocessing pipeline."""

        num_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("standardize", StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessing = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
            ("geo", ClusterSimilarity(
                n_clusters=self.config.n_clusters,
                gamma=self.config.gamma,
                random_state=self.config.random_state
            ), ["latitude", "longitude"]),
        ])

        return preprocessing

    def transform_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, TransformationResult]:
        """Apply feature engineering transformations."""
        try:
            logger.info("Starting feature engineering transformations...")

            target_column = "median_house_value"
            if target_column not in df.columns:
                raise KeyError(f"Target column '{target_column}' missing from dataset")

            y = df[target_column]
            X = df.drop(columns=[target_column])

            num_attribs = [
                "longitude", "latitude", "housing_median_age", "total_rooms",
                "total_bedrooms", "population", "households", "median_income"
            ]
            cat_attribs = ["ocean_proximity"]

            logger.info("Creating preprocessing pipeline...")
            self.preprocessing_pipeline = self.create_preprocessing_pipeline(
                num_attribs, cat_attribs
            )

            logger.info("Applying transformations...")
            processed_data = self.preprocessing_pipeline.fit_transform(X)

            num_features = num_attribs
            cat_features = self.preprocessing_pipeline.named_transformers_["cat"]\
                .named_steps["onehot"].get_feature_names_out(cat_attribs)
            geo_features = self.preprocessing_pipeline.named_transformers_["geo"]\
                .get_feature_names_out()

            all_features = np.concatenate([num_features, cat_features, geo_features])

            df_processed = pd.DataFrame(
                processed_data,
                columns=all_features,
                index=X.index
            )

            df_processed[target_column] = y

            result = TransformationResult(
                input_shape=(X.shape[0], X.shape[1]),
                output_shape=(df_processed.shape[0], df_processed.shape[1]),
                numerical_features=list(num_features),
                categorical_features=list(cat_features),
                geo_features=list(geo_features),
                target_column=target_column,
                features_added=len(all_features) - X.shape[1],
                rows_processed=df_processed.shape[0]
            )

            logger.info(f"Transformation complete: {X.shape} -> {df_processed.shape}")
            logger.info(f"Features added: {result.features_added}")

            return df_processed, result

        except Exception as e:
            logger.error(f"Error during transformation: {e}")
            raise

    def run(self) -> Dict[str, any]:
        """Execute complete feature engineering workflow."""
        try:
            logger.info("=" * 70)
            logger.info("FEATURE ENGINEERING WORKFLOW")
            logger.info("=" * 70)

            df = self.download_from_gcs()

            df_transformed, result = self.transform_data(df)

            gcs_uri = self.upload_to_gcs(df_transformed)

            logger.info("=" * 70)
            logger.info("FEATURE ENGINEERING COMPLETED")
            logger.info("=" * 70)
            logger.info(f"Input: {result.input_shape}")
            logger.info(f"Output: {result.output_shape}")
            logger.info(f"Features added: {result.features_added}")
            logger.info(f"GCS URI: {gcs_uri}")
            logger.info("=" * 70)

            return {
                "gcs_uri": gcs_uri,
                "result": result,
                "df_shape": df_transformed.shape
            }

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
