"""
Feature Engineering module for housing data
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

from __future__ import annotations

import io
import logging
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import wandb
from models import FeatureEngineeringConfig, TransformationResult

try:
    sys.path.insert(0, str(__file__).rsplit("/", 5)[0])
    from src.utils.colored_logger import setup_colored_logger

    logger = setup_colored_logger(__name__)
except (ImportError, Exception):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """Custom transformer for geographical clustering."""

    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        # NOTE: `init="random"` avoids numerical issues observed with k-means++ on this stack.
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            init="random",
            n_init=10,
            random_state=self.random_state,
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

    @staticmethod
    def _get_geo_features(df: pd.DataFrame) -> np.ndarray:
        """Extract validated geographic features for clustering."""
        geo_features = df[["latitude", "longitude"]].to_numpy(dtype=np.float64, copy=True)
        non_finite_mask = ~np.isfinite(geo_features).all(axis=1)
        if non_finite_mask.any():
            raise ValueError(
                f"Found {int(non_finite_mask.sum())} non-finite rows in latitude/longitude"
            )
        return geo_features

    def _init_gcs_client(self) -> None:
        """Initialize GCS client."""
        try:
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
        """Download data from GCS."""
        try:
            blob_path = self.config.gcs_input_path
            logger.info(f"Downloading from GCS: gs://{self.config.bucket_name}/{blob_path}")

            blob = self.bucket.blob(blob_path)
            content = blob.download_as_bytes()

            # Load DataFrame (supports CSV and Parquet)
            if blob_path.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(content))
            else:
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

            # Convert to bytes (supports CSV and Parquet)
            data_buffer = io.BytesIO()
            if blob_path.endswith(".parquet"):
                df.to_parquet(data_buffer, index=False, engine="pyarrow")
            else:
                df.to_csv(data_buffer, index=False)
            data_buffer.seek(0)

            blob = self.bucket.blob(blob_path)
            content_type = (
                "application/octet-stream" if blob_path.endswith(".parquet") else "text/csv"
            )
            blob.upload_from_file(data_buffer, content_type=content_type)

            gcs_uri = f"gs://{self.config.bucket_name}/{blob_path}"
            logger.info(f"Uploaded to: {gcs_uri}")

            return gcs_uri

        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")
            raise

    def create_preprocessing_pipeline(
        self, num_attribs: List[str], cat_attribs: List[str]
    ) -> ColumnTransformer:
        """Create sklearn preprocessing pipeline."""

        num_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("standardize", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessing = ColumnTransformer(
            [
                ("num", num_pipeline, num_attribs),
                ("cat", cat_pipeline, cat_attribs),
                (
                    "geo",
                    ClusterSimilarity(
                        n_clusters=self.config.n_clusters,
                        gamma=self.config.gamma,
                        random_state=self.config.random_state,
                    ),
                    ["latitude", "longitude"],
                ),
            ]
        )

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
                "longitude",
                "latitude",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "median_income",
            ]
            cat_attribs = ["ocean_proximity"]

            logger.info("Creating preprocessing pipeline...")
            self.preprocessing_pipeline = self.create_preprocessing_pipeline(
                num_attribs, cat_attribs
            )

            logger.info("Applying transformations...")
            processed_data = self.preprocessing_pipeline.fit_transform(X)

            num_features = num_attribs
            cat_features = (
                self.preprocessing_pipeline.named_transformers_["cat"]
                .named_steps["onehot"]
                .get_feature_names_out(cat_attribs)
            )
            geo_features = self.preprocessing_pipeline.named_transformers_[
                "geo"
            ].get_feature_names_out()

            all_features = np.concatenate([num_features, cat_features, geo_features])

            df_processed = pd.DataFrame(processed_data, columns=all_features, index=X.index)

            df_processed[target_column] = y

            result = TransformationResult(
                input_shape=(X.shape[0], X.shape[1]),
                output_shape=(df_processed.shape[0], df_processed.shape[1]),
                numerical_features=list(num_features),
                categorical_features=list(cat_features),
                geo_features=list(geo_features),
                target_column=target_column,
                features_added=len(all_features) - X.shape[1],
                rows_processed=df_processed.shape[0],
            )

            logger.info(f"Transformation complete: {X.shape} -> {df_processed.shape}")
            logger.info(f"Features added: {result.features_added}")

            return df_processed, result

        except Exception as e:
            logger.error(f"Error during transformation: {e}")
            raise

    def optimize_n_clusters(
        self, X: pd.DataFrame, min_clusters: int = 2, max_clusters: int = 20
    ) -> Tuple[int, Dict]:
        """
        Find optimal number of clusters using inertia + Calinski-Harabasz score.

        Returns:
            Tuple of (optimal_n_clusters, metrics_dict)
        """
        logger.info("=" * 70)
        logger.info("OPTIMIZING NUMBER OF CLUSTERS")
        logger.info("=" * 70)
        logger.info(f"Requested range: {min_clusters} to {max_clusters} clusters")

        geo_features = self._get_geo_features(X)
        n_samples = geo_features.shape[0]
        unique_points = np.unique(geo_features, axis=0).shape[0]

        effective_max_clusters = min(max_clusters, n_samples - 1, unique_points)
        if effective_max_clusters < min_clusters:
            min_clusters = 2
        if effective_max_clusters < min_clusters:
            raise ValueError(
                "Not enough unique geographic points to optimize clusters. "
                f"unique_points={unique_points}, min_clusters={min_clusters}"
            )

        logger.info(
            "Effective range: %s to %s clusters (n_samples=%s, unique_points=%s)",
            min_clusters,
            effective_max_clusters,
            n_samples,
            unique_points,
        )

        inertias = []
        calinski_scores = []
        cluster_range = range(min_clusters, effective_max_clusters + 1)

        max_metric_samples = 5000
        sample_indices = None
        if n_samples > max_metric_samples:
            rng = np.random.default_rng(self.config.random_state)
            sample_indices = rng.choice(n_samples, size=max_metric_samples, replace=False)
            logger.info(
                "Using sampled metrics on %s/%s rows for Calinski-Harabasz",
                max_metric_samples,
                n_samples,
            )

        for n in cluster_range:
            kmeans = KMeans(
                n_clusters=n,
                init="random",
                n_init=10,
                random_state=self.config.random_state,
            )
            labels = kmeans.fit_predict(geo_features)

            if sample_indices is not None:
                geo_for_metrics = geo_features[sample_indices]
                labels_for_metrics = labels[sample_indices]
            else:
                geo_for_metrics = geo_features
                labels_for_metrics = labels

            unique_labels = np.unique(labels_for_metrics)
            if len(unique_labels) < 2:
                current_calinski = float("-inf")
            else:
                current_calinski = float(
                    calinski_harabasz_score(geo_for_metrics, labels_for_metrics)
                )

            inertias.append(kmeans.inertia_)
            calinski_scores.append(current_calinski)

            logger.info(
                f"  n={n:2d} | Inertia: {kmeans.inertia_:.2f} | "
                f"Calinski-Harabasz: {calinski_scores[-1]:.2f}"
            )

        optimal_idx = np.argmax(calinski_scores)
        optimal_n_clusters = cluster_range[optimal_idx]

        logger.info("=" * 70)
        logger.info(f"OPTIMAL N_CLUSTERS: {optimal_n_clusters}")
        logger.info(f"  Best Calinski-Harabasz Score: {calinski_scores[optimal_idx]:.2f}")
        logger.info("=" * 70)

        metrics = {
            "cluster_range": list(cluster_range),
            "inertias": inertias,
            "calinski_scores": calinski_scores,
            "optimal_n_clusters": optimal_n_clusters,
            "best_calinski": calinski_scores[optimal_idx],
        }

        return optimal_n_clusters, metrics

    def optimize_gamma(
        self, X: pd.DataFrame, n_clusters: int, gamma_range: List[float] = None
    ) -> Tuple[float, Dict]:
        """
        Find optimal gamma value by testing different values.

        Returns:
            Tuple of (optimal_gamma, metrics_dict)
        """
        if gamma_range is None:
            gamma_range = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

        logger.info("=" * 70)
        logger.info("OPTIMIZING GAMMA VALUE")
        logger.info("=" * 70)
        logger.info(f"Testing gamma values: {gamma_range}")
        logger.info(f"Using n_clusters: {n_clusters}")

        geo_features = self._get_geo_features(X)

        variances = []

        for gamma in gamma_range:
            cluster_sim = ClusterSimilarity(
                n_clusters=n_clusters, gamma=gamma, random_state=self.config.random_state
            )
            cluster_sim.fit(geo_features)
            labels = cluster_sim.transform(geo_features).flatten()

            variance = np.var(labels)
            variances.append(variance)

            logger.info(f"  gamma={gamma:.3f} | Variance: {variance:.4f}")

        optimal_idx = np.argmax(variances)
        optimal_gamma = gamma_range[optimal_idx]

        logger.info("=" * 70)
        logger.info(f"OPTIMAL GAMMA: {optimal_gamma}")
        logger.info(f"  Best Variance: {variances[optimal_idx]:.4f}")
        logger.info("=" * 70)

        metrics = {
            "gamma_range": gamma_range,
            "variances": variances,
            "optimal_gamma": optimal_gamma,
            "best_variance": variances[optimal_idx],
        }

        return optimal_gamma, metrics

    def create_optimization_plots(
        self, cluster_metrics: Dict, gamma_metrics: Dict
    ) -> Dict[str, plt.Figure]:
        """Create visualization plots for hyperparameter optimization."""
        figures = {}

        # 1. Elbow plot for clusters
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(
            cluster_metrics["cluster_range"],
            cluster_metrics["inertias"],
            "bo-",
            linewidth=2,
            markersize=8,
        )
        ax1.set_xlabel("Number of Clusters", fontsize=12)
        ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
        ax1.set_title("Elbow Method For Optimal k", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.axvline(
            x=cluster_metrics["optimal_n_clusters"],
            color="r",
            linestyle="--",
            label=f"Optimal k={cluster_metrics['optimal_n_clusters']}",
        )
        ax1.legend()

        # 2. Calinski-Harabasz scores
        ax2.plot(
            cluster_metrics["cluster_range"],
            cluster_metrics["calinski_scores"],
            "go-",
            linewidth=2,
            markersize=8,
        )
        ax2.set_xlabel("Number of Clusters", fontsize=12)
        ax2.set_ylabel("Calinski-Harabasz Score", fontsize=12)
        ax2.set_title("Calinski-Harabasz vs Number of Clusters", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.axvline(
            x=cluster_metrics["optimal_n_clusters"],
            color="r",
            linestyle="--",
            label=f"Optimal k={cluster_metrics['optimal_n_clusters']}",
        )
        ax2.legend()

        plt.tight_layout()
        figures["cluster_optimization"] = fig1

        # 3. Gamma optimization
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            gamma_metrics["gamma_range"],
            gamma_metrics["variances"],
            "mo-",
            linewidth=2,
            markersize=8,
        )
        ax.set_xlabel("Gamma Value", fontsize=12)
        ax.set_ylabel("Label Variance", fontsize=12)
        ax.set_title("Gamma Optimization", fontsize=14, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.axvline(
            x=gamma_metrics["optimal_gamma"],
            color="r",
            linestyle="--",
            label=f"Optimal gamma={gamma_metrics['optimal_gamma']:.3f}",
        )
        ax.legend()

        plt.tight_layout()
        figures["gamma_optimization"] = fig2

        logger.info("Created optimization visualizations")

        return figures

    def log_optimization_to_wandb(
        self, cluster_metrics: Dict, gamma_metrics: Dict, figures: Dict[str, plt.Figure]
    ) -> None:
        """Log optimization results and plots to W&B."""
        logger.info("Logging optimization results to W&B...")

        # Log metrics
        wandb.log(
            {
                "optimization/optimal_n_clusters": cluster_metrics["optimal_n_clusters"],
                "optimization/best_cluster_quality_score": cluster_metrics["best_calinski"],
                "optimization/optimal_gamma": gamma_metrics["optimal_gamma"],
                "optimization/best_variance": gamma_metrics["best_variance"],
            }
        )

        # Log detailed metrics table
        cluster_table = wandb.Table(
            columns=["n_clusters", "inertia", "calinski_harabasz_score"],
            data=list(
                zip(
                    cluster_metrics["cluster_range"],
                    cluster_metrics["inertias"],
                    cluster_metrics["calinski_scores"],
                )
            ),
        )
        wandb.log({"optimization/cluster_metrics": cluster_table})

        gamma_table = wandb.Table(
            columns=["gamma", "variance"],
            data=list(zip(gamma_metrics["gamma_range"], gamma_metrics["variances"])),
        )
        wandb.log({"optimization/gamma_metrics": gamma_table})

        # Log plots
        for plot_name, fig in figures.items():
            wandb.log({f"optimization/{plot_name}": wandb.Image(fig)})
            plt.close(fig)

        logger.info("Optimization results logged to W&B successfully")

    def calculate_cluster_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate cluster quality score for current configuration.
        Uses Calinski-Harabasz to avoid unstable pairwise-distance warnings.
        """
        geo_features = self._get_geo_features(df)
        kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            init="random",
            n_init=10,
            random_state=self.config.random_state,
        )
        labels = kmeans.fit_predict(geo_features)
        score = calinski_harabasz_score(geo_features, labels)

        logger.info(
            "Cluster quality score (Calinski-Harabasz) for n_clusters=%s: %.2f",
            self.config.n_clusters,
            score,
        )
        return score

    def calculate_silhouette_score(self, df: pd.DataFrame) -> float:
        """
        Backward-compatible alias for old method name.
        """
        return self.calculate_cluster_quality_score(df)

    def run(self, optimize_hyperparams: bool = True) -> Dict[str, any]:
        """Execute complete feature engineering workflow."""
        try:
            logger.info("=" * 70)
            logger.info("FEATURE ENGINEERING WORKFLOW")
            logger.info("=" * 70)

            df = self.download_from_gcs()

            cluster_metrics = None
            gamma_metrics = None
            optimization_figures = None
            cluster_quality_score = None

            # Hyperparameter optimization
            if optimize_hyperparams:
                logger.info("\nPerforming hyperparameter optimization...")

                # Optimize n_clusters
                optimal_n_clusters, cluster_metrics = self.optimize_n_clusters(
                    df, min_clusters=2, max_clusters=20
                )

                # Optimize gamma
                optimal_gamma, gamma_metrics = self.optimize_gamma(
                    df, n_clusters=optimal_n_clusters
                )

                # Update config with optimal values
                logger.info("\nUpdating configuration with optimal hyperparameters...")
                logger.info(f"  Original n_clusters: {self.config.n_clusters}")
                logger.info(f"  Optimal n_clusters: {optimal_n_clusters}")
                logger.info(f"  Original gamma: {self.config.gamma}")
                logger.info(f"  Optimal gamma: {optimal_gamma}")

                self.config.n_clusters = optimal_n_clusters
                self.config.gamma = optimal_gamma

                # Create optimization visualizations
                optimization_figures = self.create_optimization_plots(
                    cluster_metrics, gamma_metrics
                )

                # Log to W&B
                self.log_optimization_to_wandb(cluster_metrics, gamma_metrics, optimization_figures)
            else:
                # For sweep: calculate cluster quality score for current params
                logger.info("\nCalculating cluster quality score for current configuration...")
                cluster_quality_score = self.calculate_cluster_quality_score(df)

            df_transformed, result = self.transform_data(df)

            gcs_uri = self.upload_to_gcs(df_transformed)

            logger.info("=" * 70)
            logger.info("FEATURE ENGINEERING COMPLETED")
            logger.info("=" * 70)
            logger.info(f"Input: {result.input_shape}")
            logger.info(f"Output: {result.output_shape}")
            logger.info(f"Features added: {result.features_added}")
            if optimize_hyperparams:
                logger.info(f"Optimal n_clusters: {optimal_n_clusters}")
                logger.info(f"Optimal gamma: {optimal_gamma}")
            logger.info(f"GCS URI: {gcs_uri}")
            logger.info("=" * 70)

            return {
                "gcs_uri": gcs_uri,
                "result": result,
                "df_shape": df_transformed.shape,
                "cluster_metrics": cluster_metrics,
                "gamma_metrics": gamma_metrics,
                "optimization_enabled": optimize_hyperparams,
                "cluster_quality_score": cluster_quality_score,
                # Backward-compat key name used by older callers.
                "silhouette_score": cluster_quality_score,
            }

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
