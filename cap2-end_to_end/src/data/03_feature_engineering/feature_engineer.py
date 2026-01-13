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
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score
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

    def optimize_n_clusters(
        self,
        X: pd.DataFrame,
        min_clusters: int = 2,
        max_clusters: int = 20
    ) -> Tuple[int, Dict]:
        """
        Find optimal number of clusters using elbow method and silhouette score.

        Returns:
            Tuple of (optimal_n_clusters, metrics_dict)
        """
        logger.info("=" * 70)
        logger.info("OPTIMIZING NUMBER OF CLUSTERS")
        logger.info("=" * 70)
        logger.info(f"Testing range: {min_clusters} to {max_clusters} clusters")

        geo_features = X[["latitude", "longitude"]].values

        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        cluster_range = range(min_clusters, max_clusters + 1)

        for n in cluster_range:
            kmeans = KMeans(n_clusters=n, n_init=10, random_state=self.config.random_state)
            labels = kmeans.fit_predict(geo_features)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(geo_features, labels))
            davies_bouldin_scores.append(davies_bouldin_score(geo_features, labels))

            logger.info(f"  n={n:2d} | Inertia: {kmeans.inertia_:.2f} | "
                       f"Silhouette: {silhouette_scores[-1]:.4f} | "
                       f"Davies-Bouldin: {davies_bouldin_scores[-1]:.4f}")

        optimal_idx = np.argmax(silhouette_scores)
        optimal_n_clusters = cluster_range[optimal_idx]

        logger.info("=" * 70)
        logger.info(f"OPTIMAL N_CLUSTERS: {optimal_n_clusters}")
        logger.info(f"  Best Silhouette Score: {silhouette_scores[optimal_idx]:.4f}")
        logger.info("=" * 70)

        metrics = {
            "cluster_range": list(cluster_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "davies_bouldin_scores": davies_bouldin_scores,
            "optimal_n_clusters": optimal_n_clusters,
            "best_silhouette": silhouette_scores[optimal_idx]
        }

        return optimal_n_clusters, metrics

    def optimize_gamma(
        self,
        X: pd.DataFrame,
        n_clusters: int,
        gamma_range: List[float] = None
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

        geo_features = X[["latitude", "longitude"]].values

        variances = []

        for gamma in gamma_range:
            cluster_sim = ClusterSimilarity(
                n_clusters=n_clusters,
                gamma=gamma,
                random_state=self.config.random_state
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
            "best_variance": variances[optimal_idx]
        }

        return optimal_gamma, metrics

    def create_optimization_plots(
        self,
        cluster_metrics: Dict,
        gamma_metrics: Dict
    ) -> Dict[str, plt.Figure]:
        """Create visualization plots for hyperparameter optimization."""
        figures = {}

        # 1. Elbow plot for clusters
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(cluster_metrics["cluster_range"],
                cluster_metrics["inertias"],
                'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters', fontsize=12)
        ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        ax1.set_title('Elbow Method For Optimal k', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=cluster_metrics["optimal_n_clusters"],
                   color='r', linestyle='--', label=f'Optimal k={cluster_metrics["optimal_n_clusters"]}')
        ax1.legend()

        # 2. Silhouette scores
        ax2.plot(cluster_metrics["cluster_range"],
                cluster_metrics["silhouette_scores"],
                'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=cluster_metrics["optimal_n_clusters"],
                   color='r', linestyle='--', label=f'Optimal k={cluster_metrics["optimal_n_clusters"]}')
        ax2.legend()

        plt.tight_layout()
        figures["cluster_optimization"] = fig1

        # 3. Gamma optimization
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(gamma_metrics["gamma_range"],
               gamma_metrics["variances"],
               'mo-', linewidth=2, markersize=8)
        ax.set_xlabel('Gamma Value', fontsize=12)
        ax.set_ylabel('Label Variance', fontsize=12)
        ax.set_title('Gamma Optimization', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=gamma_metrics["optimal_gamma"],
                  color='r', linestyle='--',
                  label=f'Optimal gamma={gamma_metrics["optimal_gamma"]:.3f}')
        ax.legend()

        plt.tight_layout()
        figures["gamma_optimization"] = fig2

        # 4. Davies-Bouldin Index
        fig3, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cluster_metrics["cluster_range"],
               cluster_metrics["davies_bouldin_scores"],
               'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Davies-Bouldin Index (lower is better)', fontsize=12)
        ax.set_title('Davies-Bouldin Index vs Number of Clusters', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=cluster_metrics["optimal_n_clusters"],
                  color='g', linestyle='--', label=f'Optimal k={cluster_metrics["optimal_n_clusters"]}')
        ax.legend()

        plt.tight_layout()
        figures["davies_bouldin"] = fig3

        logger.info("Created optimization visualizations")

        return figures

    def log_optimization_to_wandb(
        self,
        cluster_metrics: Dict,
        gamma_metrics: Dict,
        figures: Dict[str, plt.Figure]
    ) -> None:
        """Log optimization results and plots to W&B."""
        logger.info("Logging optimization results to W&B...")

        # Log metrics
        wandb.log({
            "optimization/optimal_n_clusters": cluster_metrics["optimal_n_clusters"],
            "optimization/best_silhouette_score": cluster_metrics["best_silhouette"],
            "optimization/optimal_gamma": gamma_metrics["optimal_gamma"],
            "optimization/best_variance": gamma_metrics["best_variance"],
        })

        # Log detailed metrics table
        cluster_table = wandb.Table(
            columns=["n_clusters", "inertia", "silhouette_score", "davies_bouldin_score"],
            data=list(zip(
                cluster_metrics["cluster_range"],
                cluster_metrics["inertias"],
                cluster_metrics["silhouette_scores"],
                cluster_metrics["davies_bouldin_scores"]
            ))
        )
        wandb.log({"optimization/cluster_metrics": cluster_table})

        gamma_table = wandb.Table(
            columns=["gamma", "variance"],
            data=list(zip(
                gamma_metrics["gamma_range"],
                gamma_metrics["variances"]
            ))
        )
        wandb.log({"optimization/gamma_metrics": gamma_table})

        # Log plots
        for plot_name, fig in figures.items():
            wandb.log({f"optimization/{plot_name}": wandb.Image(fig)})
            plt.close(fig)

        logger.info("Optimization results logged to W&B successfully")

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

            # Hyperparameter optimization
            if optimize_hyperparams:
                logger.info("\nPerforming hyperparameter optimization...")

                # Optimize n_clusters
                optimal_n_clusters, cluster_metrics = self.optimize_n_clusters(
                    df,
                    min_clusters=2,
                    max_clusters=20
                )

                # Optimize gamma
                optimal_gamma, gamma_metrics = self.optimize_gamma(
                    df,
                    n_clusters=optimal_n_clusters
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
                    cluster_metrics,
                    gamma_metrics
                )

                # Log to W&B
                self.log_optimization_to_wandb(
                    cluster_metrics,
                    gamma_metrics,
                    optimization_figures
                )

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
                "optimization_enabled": optimize_hyperparams
            }

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
