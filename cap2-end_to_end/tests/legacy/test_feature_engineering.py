"""
Unit tests for feature engineering module
Author: Carlos Daniel Jiménez
Date: 2026-01-13
"""

import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "data" / "03_feature_engineering"))

from feature_engineer import ClusterSimilarity, FeatureEngineer

from models import FeatureEngineeringConfig, TransformationResult


class TestFeatureEngineeringConfig:
    """Test FeatureEngineeringConfig model."""

    def test_config_creation_success(self):
        """Test successful config creation with valid parameters."""
        config = FeatureEngineeringConfig(
            input_artifact_name="test_artifact:latest",
            gcs_input_path="data/input.csv",
            gcs_output_path="data/output.csv",
            artifact_name="test_output",
            bucket_name="test-bucket",
            wandb_project="test-project",
            n_clusters=10,
            gamma=0.1,
            random_state=42,
        )

        assert config.input_artifact_name == "test_artifact:latest"
        assert config.n_clusters == 10
        assert config.gamma == 0.1
        assert config.random_state == 42

    def test_config_validation_n_clusters_too_small(self):
        """Test validation fails for n_clusters < 2."""
        with pytest.raises(ValueError, match="n_clusters must be between 2 and 50"):
            FeatureEngineeringConfig(
                input_artifact_name="test",
                gcs_input_path="data/input.csv",
                gcs_output_path="data/output.csv",
                artifact_name="test",
                bucket_name="test-bucket",
                wandb_project="test-project",
                n_clusters=1,
            )

    def test_config_validation_n_clusters_too_large(self):
        """Test validation fails for n_clusters > 50."""
        with pytest.raises(ValueError, match="n_clusters must be between 2 and 50"):
            FeatureEngineeringConfig(
                input_artifact_name="test",
                gcs_input_path="data/input.csv",
                gcs_output_path="data/output.csv",
                artifact_name="test",
                bucket_name="test-bucket",
                wandb_project="test-project",
                n_clusters=51,
            )

    def test_config_validation_gamma_negative(self):
        """Test validation fails for negative gamma."""
        with pytest.raises(ValueError, match="gamma must be positive"):
            FeatureEngineeringConfig(
                input_artifact_name="test",
                gcs_input_path="data/input.csv",
                gcs_output_path="data/output.csv",
                artifact_name="test",
                bucket_name="test-bucket",
                wandb_project="test-project",
                gamma=-0.1,
            )


class TestClusterSimilarity:
    """Test ClusterSimilarity transformer."""

    def test_cluster_similarity_fit(self):
        """Test fitting ClusterSimilarity transformer."""
        X = np.random.rand(100, 2)
        transformer = ClusterSimilarity(n_clusters=5, random_state=42)

        result = transformer.fit(X)

        assert result is transformer
        assert hasattr(transformer, "kmeans_")
        assert transformer.kmeans_.n_clusters == 5

    def test_cluster_similarity_transform(self):
        """Test transforming data with ClusterSimilarity."""
        X = np.random.rand(100, 2)
        transformer = ClusterSimilarity(n_clusters=5, random_state=42)

        transformer.fit(X)
        transformed = transformer.transform(X)

        assert transformed.shape == (100, 1)
        assert np.all(transformed >= 0)
        assert np.all(transformed < 5)

    def test_cluster_similarity_get_feature_names(self):
        """Test getting feature names from transformer."""
        transformer = ClusterSimilarity(n_clusters=5)
        names = transformer.get_feature_names_out()

        assert names == ["cluster_label"]


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return FeatureEngineeringConfig(
            input_artifact_name="test_artifact:latest",
            gcs_input_path="data/input.csv",
            gcs_output_path="data/output.csv",
            artifact_name="test_output",
            bucket_name="test-bucket",
            wandb_project="test-project",
            n_clusters=10,
            gamma=0.1,
            random_state=42,
        )

    @pytest.fixture
    def sample_housing_data(self):
        """Create sample housing DataFrame."""
        np.random.seed(42)
        n_samples = 100

        return pd.DataFrame(
            {
                "longitude": np.random.uniform(-125, -114, n_samples),
                "latitude": np.random.uniform(32, 42, n_samples),
                "housing_median_age": np.random.uniform(1, 52, n_samples),
                "total_rooms": np.random.uniform(100, 3000, n_samples),
                "total_bedrooms": np.random.uniform(50, 1500, n_samples),
                "population": np.random.uniform(100, 3000, n_samples),
                "households": np.random.uniform(50, 1500, n_samples),
                "median_income": np.random.uniform(0.5, 15, n_samples),
                "ocean_proximity": np.random.choice(["NEAR BAY", "INLAND", "<1H OCEAN"], n_samples),
                "median_house_value": np.random.uniform(50000, 500000, n_samples),
            }
        )

    @pytest.fixture
    def mock_gcs_client(self):
        """Create mock GCS client."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_bucket.exists.return_value = True
        mock_client.bucket.return_value = mock_bucket

        return {"client": mock_client, "bucket": mock_bucket}

    def test_init_success(self, sample_config, mock_gcs_client, monkeypatch):
        """Test successful initialization."""
        monkeypatch.setattr("feature_engineer.storage.Client", lambda: mock_gcs_client["client"])

        engineer = FeatureEngineer(sample_config)

        assert engineer.config == sample_config
        assert engineer.storage_client is not None
        assert engineer.bucket is not None

    def test_init_bucket_not_exists(self, sample_config, mock_gcs_client, monkeypatch):
        """Test initialization fails when bucket doesn't exist."""
        mock_gcs_client["bucket"].exists.return_value = False
        monkeypatch.setattr("feature_engineer.storage.Client", lambda: mock_gcs_client["client"])

        with pytest.raises(RuntimeError, match="GCS is required"):
            FeatureEngineer(sample_config)

    def test_download_from_gcs_success(
        self, sample_config, sample_housing_data, mock_gcs_client, monkeypatch
    ):
        """Test successful download from GCS."""
        monkeypatch.setattr("feature_engineer.storage.Client", lambda: mock_gcs_client["client"])

        csv_buffer = BytesIO()
        sample_housing_data.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue()

        mock_blob = Mock()
        mock_blob.download_as_bytes.return_value = csv_bytes
        mock_gcs_client["bucket"].blob.return_value = mock_blob

        engineer = FeatureEngineer(sample_config)
        df = engineer.download_from_gcs()

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 100
        assert "median_house_value" in df.columns

    def test_upload_to_gcs_success(
        self, sample_config, sample_housing_data, mock_gcs_client, monkeypatch
    ):
        """Test successful upload to GCS."""
        monkeypatch.setattr("feature_engineer.storage.Client", lambda: mock_gcs_client["client"])

        mock_blob = Mock()
        mock_gcs_client["bucket"].blob.return_value = mock_blob

        engineer = FeatureEngineer(sample_config)
        gcs_uri = engineer.upload_to_gcs(sample_housing_data)

        assert gcs_uri == f"gs://{sample_config.bucket_name}/{sample_config.gcs_output_path}"
        mock_blob.upload_from_file.assert_called_once()

    def test_create_preprocessing_pipeline(self, sample_config, mock_gcs_client, monkeypatch):
        """Test creating preprocessing pipeline."""
        monkeypatch.setattr("feature_engineer.storage.Client", lambda: mock_gcs_client["client"])

        engineer = FeatureEngineer(sample_config)

        num_attribs = ["longitude", "latitude", "housing_median_age"]
        cat_attribs = ["ocean_proximity"]

        pipeline = engineer.create_preprocessing_pipeline(num_attribs, cat_attribs)

        assert pipeline is not None
        assert hasattr(pipeline, "transformers")
        assert len(pipeline.transformers) == 3  # num, cat, geo

    def test_transform_data_success(
        self, sample_config, sample_housing_data, mock_gcs_client, monkeypatch
    ):
        """Test successful data transformation."""
        monkeypatch.setattr("feature_engineer.storage.Client", lambda: mock_gcs_client["client"])

        engineer = FeatureEngineer(sample_config)
        df_transformed, result = engineer.transform_data(sample_housing_data)

        assert isinstance(df_transformed, pd.DataFrame)
        assert isinstance(result, TransformationResult)
        assert df_transformed.shape[0] == sample_housing_data.shape[0]
        assert df_transformed.shape[1] > sample_housing_data.shape[1]
        assert "median_house_value" in df_transformed.columns
        assert result.features_added > 0

    def test_transform_data_missing_target(
        self, sample_config, sample_housing_data, mock_gcs_client, monkeypatch
    ):
        """Test transformation fails when target column is missing."""
        monkeypatch.setattr("feature_engineer.storage.Client", lambda: mock_gcs_client["client"])

        df_no_target = sample_housing_data.drop(columns=["median_house_value"])
        engineer = FeatureEngineer(sample_config)

        with pytest.raises(KeyError, match="Target column"):
            engineer.transform_data(df_no_target)


class TestTransformationResult:
    """Test TransformationResult model."""

    def test_transformation_result_creation(self):
        """Test creating TransformationResult."""
        result = TransformationResult(
            input_shape=(100, 10),
            output_shape=(100, 20),
            numerical_features=["feat1", "feat2"],
            categorical_features=["cat1", "cat2"],
            geo_features=["geo1"],
            target_column="target",
            features_added=10,
            rows_processed=100,
        )

        assert result.input_shape == (100, 10)
        assert result.output_shape == (100, 20)
        assert result.features_added == 10
        assert len(result.numerical_features) == 2
