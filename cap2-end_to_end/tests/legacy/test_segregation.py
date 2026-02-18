"""
Unit tests for data segregation module
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
sys.path.insert(0, str(project_root / "src" / "data" / "04_segregation"))

from segregator import DataSegregator

from models import SegregationConfig, SegregationResult


class TestSegregationConfig:
    """Test SegregationConfig model."""

    def test_config_creation_success(self):
        """Test successful config creation with valid parameters."""
        config = SegregationConfig(
            input_artifact_name="test_artifact:latest",
            gcs_input_path="data/input.csv",
            gcs_train_output_path="data/train.csv",
            gcs_test_output_path="data/test.csv",
            artifact_root="test_data",
            artifact_type="segregated_data",
            artifact_description="Test data split",
            bucket_name="test-bucket",
            wandb_project="test-project",
            test_size=0.2,
            random_state=42,
            target_column="target",
        )

        assert config.input_artifact_name == "test_artifact:latest"
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.target_column == "target"

    def test_config_validation_test_size_too_small(self):
        """Test validation fails for test_size <= 0."""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            SegregationConfig(
                input_artifact_name="test",
                gcs_input_path="data/input.csv",
                gcs_train_output_path="data/train.csv",
                gcs_test_output_path="data/test.csv",
                artifact_root="test_data",
                bucket_name="test-bucket",
                wandb_project="test-project",
                test_size=0.0,
            )

    def test_config_validation_test_size_too_large(self):
        """Test validation fails for test_size >= 1."""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            SegregationConfig(
                input_artifact_name="test",
                gcs_input_path="data/input.csv",
                gcs_train_output_path="data/train.csv",
                gcs_test_output_path="data/test.csv",
                artifact_root="test_data",
                bucket_name="test-bucket",
                wandb_project="test-project",
                test_size=1.0,
            )

    def test_config_validation_random_state_negative(self):
        """Test validation fails for negative random_state."""
        with pytest.raises(ValueError, match="random_state must be non-negative"):
            SegregationConfig(
                input_artifact_name="test",
                gcs_input_path="data/input.csv",
                gcs_train_output_path="data/train.csv",
                gcs_test_output_path="data/test.csv",
                artifact_root="test_data",
                bucket_name="test-bucket",
                wandb_project="test-project",
                random_state=-1,
            )

    def test_config_default_values(self):
        """Test default values are applied correctly."""
        config = SegregationConfig(
            input_artifact_name="test",
            gcs_input_path="data/input.csv",
            gcs_train_output_path="data/train.csv",
            gcs_test_output_path="data/test.csv",
            artifact_root="test_data",
            bucket_name="test-bucket",
            wandb_project="test-project",
        )

        assert config.artifact_type == "segregated_data"
        assert config.artifact_description == ""
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.target_column == "median_house_value"


class TestDataSegregator:
    """Test DataSegregator class."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return SegregationConfig(
            input_artifact_name="test_artifact:latest",
            gcs_input_path="data/input.csv",
            gcs_train_output_path="data/train.csv",
            gcs_test_output_path="data/test.csv",
            artifact_root="test_data",
            artifact_type="segregated_data",
            artifact_description="Test data split",
            bucket_name="test-bucket",
            wandb_project="test-project",
            test_size=0.2,
            random_state=42,
            target_column="target",
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n_samples = 100

        return pd.DataFrame(
            {
                "feature1": np.random.uniform(0, 100, n_samples),
                "feature2": np.random.uniform(0, 100, n_samples),
                "feature3": np.random.uniform(0, 100, n_samples),
                "target": np.random.uniform(50000, 500000, n_samples),
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
        monkeypatch.setattr("segregator.storage.Client", lambda: mock_gcs_client["client"])

        segregator = DataSegregator(sample_config)

        assert segregator.config == sample_config
        assert segregator.storage_client is not None
        assert segregator.bucket is not None

    def test_init_bucket_not_exists(self, sample_config, mock_gcs_client, monkeypatch):
        """Test initialization fails when bucket doesn't exist."""
        mock_gcs_client["bucket"].exists.return_value = False
        monkeypatch.setattr("segregator.storage.Client", lambda: mock_gcs_client["client"])

        with pytest.raises(RuntimeError, match="GCS is required"):
            DataSegregator(sample_config)

    def test_download_from_gcs_success(
        self, sample_config, sample_data, mock_gcs_client, monkeypatch
    ):
        """Test successful download from GCS."""
        monkeypatch.setattr("segregator.storage.Client", lambda: mock_gcs_client["client"])

        csv_buffer = BytesIO()
        sample_data.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue()

        mock_blob = Mock()
        mock_blob.download_as_bytes.return_value = csv_bytes
        mock_gcs_client["bucket"].blob.return_value = mock_blob

        segregator = DataSegregator(sample_config)
        df = segregator.download_from_gcs()

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 100
        assert "target" in df.columns

    def test_upload_to_gcs_success(self, sample_config, sample_data, mock_gcs_client, monkeypatch):
        """Test successful upload to GCS."""
        monkeypatch.setattr("segregator.storage.Client", lambda: mock_gcs_client["client"])

        mock_blob = Mock()
        mock_gcs_client["bucket"].blob.return_value = mock_blob

        segregator = DataSegregator(sample_config)
        gcs_uri = segregator.upload_to_gcs(sample_data, "data/output.csv")

        assert gcs_uri == "gs://test-bucket/data/output.csv"
        mock_blob.upload_from_file.assert_called_once()

    def test_split_data_success(self, sample_config, sample_data, mock_gcs_client, monkeypatch):
        """Test successful data splitting."""
        monkeypatch.setattr("segregator.storage.Client", lambda: mock_gcs_client["client"])

        segregator = DataSegregator(sample_config)
        train_data, test_data = segregator.split_data(sample_data)

        assert isinstance(train_data, pd.DataFrame)
        assert isinstance(test_data, pd.DataFrame)
        assert train_data.shape[0] == 80
        assert test_data.shape[0] == 20
        assert "target" in train_data.columns
        assert "target" in test_data.columns
        assert train_data.shape[0] + test_data.shape[0] == sample_data.shape[0]

    def test_split_data_missing_target(
        self, sample_config, sample_data, mock_gcs_client, monkeypatch
    ):
        """Test splitting fails when target column is missing."""
        monkeypatch.setattr("segregator.storage.Client", lambda: mock_gcs_client["client"])

        df_no_target = sample_data.drop(columns=["target"])
        segregator = DataSegregator(sample_config)

        with pytest.raises(KeyError, match="Target column"):
            segregator.split_data(df_no_target)

    def test_split_data_preserves_columns(
        self, sample_config, sample_data, mock_gcs_client, monkeypatch
    ):
        """Test that splitting preserves all columns including target."""
        monkeypatch.setattr("segregator.storage.Client", lambda: mock_gcs_client["client"])

        segregator = DataSegregator(sample_config)
        train_data, test_data = segregator.split_data(sample_data)

        assert set(train_data.columns) == set(sample_data.columns)
        assert set(test_data.columns) == set(sample_data.columns)

    def test_split_data_random_state(
        self, sample_config, sample_data, mock_gcs_client, monkeypatch
    ):
        """Test that random_state ensures reproducibility."""
        monkeypatch.setattr("segregator.storage.Client", lambda: mock_gcs_client["client"])

        segregator1 = DataSegregator(sample_config)
        train1, test1 = segregator1.split_data(sample_data.copy())

        segregator2 = DataSegregator(sample_config)
        train2, test2 = segregator2.split_data(sample_data.copy())

        pd.testing.assert_frame_equal(train1.reset_index(drop=True), train2.reset_index(drop=True))
        pd.testing.assert_frame_equal(test1.reset_index(drop=True), test2.reset_index(drop=True))

    def test_split_data_different_test_sizes(self, sample_data, mock_gcs_client, monkeypatch):
        """Test splitting with different test_size values."""
        monkeypatch.setattr("segregator.storage.Client", lambda: mock_gcs_client["client"])

        for test_size in [0.1, 0.2, 0.3]:
            config = SegregationConfig(
                input_artifact_name="test",
                gcs_input_path="data/input.csv",
                gcs_train_output_path="data/train.csv",
                gcs_test_output_path="data/test.csv",
                artifact_root="test_data",
                bucket_name="test-bucket",
                wandb_project="test-project",
                test_size=test_size,
                target_column="target",
            )

            segregator = DataSegregator(config)
            train_data, test_data = segregator.split_data(sample_data)

            expected_test_size = int(100 * test_size)
            assert test_data.shape[0] == expected_test_size
            assert train_data.shape[0] == 100 - expected_test_size


class TestSegregationResult:
    """Test SegregationResult model."""

    def test_segregation_result_creation(self):
        """Test creating SegregationResult."""
        result = SegregationResult(
            input_shape=(100, 10),
            train_shape=(80, 10),
            test_shape=(20, 10),
            target_column="target",
            test_size=0.2,
            train_gcs_uri="gs://bucket/train.csv",
            test_gcs_uri="gs://bucket/test.csv",
            train_samples=80,
            test_samples=20,
        )

        assert result.input_shape == (100, 10)
        assert result.train_shape == (80, 10)
        assert result.test_shape == (20, 10)
        assert result.target_column == "target"
        assert result.test_size == 0.2
        assert result.train_samples == 80
        assert result.test_samples == 20

    def test_segregation_result_gcs_uris(self):
        """Test GCS URIs are stored correctly."""
        result = SegregationResult(
            input_shape=(100, 10),
            train_shape=(80, 10),
            test_shape=(20, 10),
            target_column="target",
            test_size=0.2,
            train_gcs_uri="gs://my-bucket/path/to/train.csv",
            test_gcs_uri="gs://my-bucket/path/to/test.csv",
            train_samples=80,
            test_samples=20,
        )

        assert result.train_gcs_uri.startswith("gs://")
        assert result.test_gcs_uri.startswith("gs://")
        assert "train.csv" in result.train_gcs_uri
        assert "test.csv" in result.test_gcs_uri


class TestDataSegregatorWorkflow:
    """Test complete DataSegregator workflow."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return SegregationConfig(
            input_artifact_name="test_artifact:latest",
            gcs_input_path="data/input.csv",
            gcs_train_output_path="data/train.csv",
            gcs_test_output_path="data/test.csv",
            artifact_root="test_data",
            artifact_type="segregated_data",
            artifact_description="Test data split",
            bucket_name="test-bucket",
            wandb_project="test-project",
            test_size=0.2,
            random_state=42,
            target_column="target",
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n_samples = 100

        return pd.DataFrame(
            {
                "feature1": np.random.uniform(0, 100, n_samples),
                "feature2": np.random.uniform(0, 100, n_samples),
                "feature3": np.random.uniform(0, 100, n_samples),
                "target": np.random.uniform(50000, 500000, n_samples),
            }
        )

    @pytest.fixture
    def mock_gcs_client(self, sample_data):
        """Create mock GCS client with data."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_bucket.exists.return_value = True
        mock_client.bucket.return_value = mock_bucket

        csv_buffer = BytesIO()
        sample_data.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue()

        mock_blob = Mock()
        mock_blob.download_as_bytes.return_value = csv_bytes
        mock_bucket.blob.return_value = mock_blob

        return {"client": mock_client, "bucket": mock_bucket, "blob": mock_blob}

    def test_complete_workflow(self, sample_config, mock_gcs_client, monkeypatch):
        """Test complete segregation workflow."""
        monkeypatch.setattr("segregator.storage.Client", lambda: mock_gcs_client["client"])

        segregator = DataSegregator(sample_config)
        result_data = segregator.run()

        assert "result" in result_data
        assert "train_data" in result_data
        assert "test_data" in result_data

        result = result_data["result"]
        assert isinstance(result, SegregationResult)
        assert result.input_shape == (100, 4)
        assert result.train_samples == 80
        assert result.test_samples == 20

        train_data = result_data["train_data"]
        test_data = result_data["test_data"]
        assert train_data.shape[0] == 80
        assert test_data.shape[0] == 20
        assert "target" in train_data.columns
        assert "target" in test_data.columns
