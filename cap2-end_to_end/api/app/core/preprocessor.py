"""
Preprocessor for raw housing features.
Transforms raw input to model-ready features.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class HousingPreprocessor:
    """
    Preprocesses raw housing features for model prediction.

    Transformations:
    1. One-hot encodes ocean_proximity
    2. Creates cluster_label using KMeans on lat/lon
    """

    # Expected ocean proximity categories (from training)
    OCEAN_PROXIMITY_CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

    # Expected feature order (from model.feature_names_in_)
    EXPECTED_FEATURES = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity_<1H OCEAN",
        "ocean_proximity_INLAND",
        "ocean_proximity_ISLAND",
        "ocean_proximity_NEAR BAY",
        "ocean_proximity_NEAR OCEAN",
        "cluster_label",
    ]

    def __init__(self, n_clusters: int = 10):
        """
        Initialize preprocessor.

        Args:
            n_clusters: Number of geographical clusters for KMeans
        """
        self.n_clusters = n_clusters
        self.kmeans = None
        self._init_kmeans()

    def _init_kmeans(self):
        """
        Initialize KMeans with California housing geographical clusters.

        Uses typical California housing coordinates to create clusters.
        This is an approximation but works for the API use case.
        """
        # California housing typical ranges:
        # Longitude: -124 to -114
        # Latitude: 32 to 42

        # Create representative centroids for California regions
        # These approximate the major housing markets in California
        np.random.seed(42)

        # Create synthetic data representing California's geography
        # This will create stable clusters
        n_samples = 1000
        lon_samples = np.random.uniform(-124, -114, n_samples)
        lat_samples = np.random.uniform(32, 42, n_samples)

        # Weight towards major population centers
        # LA area: ~-118, 34
        # SF area: ~-122, 37.5
        # San Diego: ~-117, 33
        # Sacramento: ~-121, 38.5

        major_centers = np.array(
            [
                [-118, 34],  # LA
                [-122, 37.5],  # SF
                [-117, 33],  # San Diego
                [-121, 38.5],  # Sacramento
                [-119, 36.5],  # Fresno area
            ]
        )

        # Add major centers multiple times for proper weighting
        lon_samples[:50] = major_centers[:, 0].repeat(10)
        lat_samples[:50] = major_centers[:, 1].repeat(10)

        X_geo = np.column_stack([lon_samples, lat_samples])

        # Fit KMeans
        # Use random init to avoid numerical warnings seen with k-means++ in this environment.
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init="random",
            n_init=10,
            random_state=42,
        )
        self.kmeans.fit(X_geo)

        logger.info(f"Initialized KMeans with {self.n_clusters} clusters")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw features to model-ready features.

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with processed features in correct order

        Raises:
            ValueError: If required columns are missing
        """
        # Validate input
        required_cols = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "ocean_proximity",
        ]

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create copy to avoid modifying original
        df_processed = df.copy()

        # 1. One-hot encode ocean_proximity
        for category in self.OCEAN_PROXIMITY_CATEGORIES:
            col_name = f"ocean_proximity_{category}"
            df_processed[col_name] = (df["ocean_proximity"] == category).astype(int)

        # 2. Create cluster_label using KMeans
        X_geo = df_processed[["longitude", "latitude"]].values
        cluster_labels = self.kmeans.predict(X_geo)
        df_processed["cluster_label"] = cluster_labels

        # 3. Select and order columns as expected by model
        df_final = df_processed[self.EXPECTED_FEATURES]

        logger.debug(f"Transformed {len(df)} rows with {len(self.EXPECTED_FEATURES)} features")

        return df_final

    def get_feature_names(self) -> List[str]:
        """Get list of feature names in correct order."""
        return self.EXPECTED_FEATURES.copy()
