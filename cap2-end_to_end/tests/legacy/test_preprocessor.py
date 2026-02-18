"""
Unit tests for DataPreprocessor class
Author: Carlos Daniel Jiménez
Date: 2025-01-13
"""

import io

import pandas as pd
import pytest
from google.api_core import exceptions as gcp_exceptions
from preprocessor import DataPreprocessor

from models import PreprocessingConfig, PreprocessingResult, PreprocessingStats


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

    def test_init_success(self, preprocessing_config_dict, mock_gcs_client, monkeypatch):
        """Test successful initialization with GCS."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        assert preprocessor.config == config
        assert preprocessor.storage_client is not None
        assert preprocessor.bucket is not None

    def test_init_bucket_not_exists(self, preprocessing_config_dict, mock_gcs_client, monkeypatch):
        """Test initialization fails when bucket doesn't exist."""
        mock_gcs_client["bucket"].exists.return_value = False
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config = PreprocessingConfig(**preprocessing_config_dict)

        with pytest.raises(RuntimeError, match="GCS is required"):
            DataPreprocessor(config)

    def test_download_from_gcs_success(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test successful download from GCS."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        # Mock blob download
        csv_buffer = io.StringIO()
        sample_housing_data.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        mock_gcs_client["blob"].exists.return_value = True
        mock_gcs_client["blob"].download_as_bytes.return_value = csv_bytes

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        df = preprocessor.download_from_gcs()

        assert len(df) == len(sample_housing_data)
        assert list(df.columns) == list(sample_housing_data.columns)
        mock_gcs_client["blob"].download_as_bytes.assert_called_once()

    def test_download_from_gcs_file_not_found(
        self, preprocessing_config_dict, mock_gcs_client, monkeypatch
    ):
        """Test download fails when file doesn't exist in GCS."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        mock_gcs_client["blob"].exists.return_value = False

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        with pytest.raises(FileNotFoundError, match="File not found in GCS"):
            preprocessor.download_from_gcs()

    def test_download_from_gcs_api_error(
        self, preprocessing_config_dict, mock_gcs_client, monkeypatch
    ):
        """Test download fails with GCS API error."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        mock_gcs_client["blob"].exists.return_value = True
        mock_gcs_client["blob"].download_as_bytes.side_effect = gcp_exceptions.GoogleAPIError(
            "Download failed"
        )

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        with pytest.raises(RuntimeError, match="Error downloading from GCS"):
            preprocessor.download_from_gcs()

    def test_get_input_stats(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test getting input statistics."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        stats = preprocessor.get_input_stats(sample_housing_data)

        assert stats["input_rows"] == len(sample_housing_data)
        assert stats["input_columns"] == len(sample_housing_data.columns)
        assert "missing_values_before" in stats
        assert "input_size_mb" in stats

    def test_handle_missing_values_drop(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test dropping rows with missing values."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config_dict = preprocessing_config_dict.copy()
        config_dict["imputation_strategy"] = "drop"
        config = PreprocessingConfig(**config_dict)
        preprocessor = DataPreprocessor(config)

        original_len = len(sample_housing_data)
        df_processed = preprocessor.handle_missing_values(sample_housing_data)

        assert len(df_processed) < original_len
        assert df_processed.isnull().sum().sum() == 0

    def test_handle_missing_values_median(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test imputation with median."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config_dict = preprocessing_config_dict.copy()
        config_dict["imputation_strategy"] = "median"
        config = PreprocessingConfig(**config_dict)
        preprocessor = DataPreprocessor(config)

        df_processed = preprocessor.handle_missing_values(sample_housing_data)

        assert len(df_processed) == len(sample_housing_data)
        assert df_processed["total_bedrooms"].isnull().sum() == 0

    def test_handle_missing_values_mean(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test imputation with mean."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config_dict = preprocessing_config_dict.copy()
        config_dict["imputation_strategy"] = "mean"
        config = PreprocessingConfig(**config_dict)
        preprocessor = DataPreprocessor(config)

        df_processed = preprocessor.handle_missing_values(sample_housing_data)

        assert len(df_processed) == len(sample_housing_data)
        assert df_processed["total_bedrooms"].isnull().sum() == 0

    def test_handle_missing_values_mode(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test imputation with mode."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config_dict = preprocessing_config_dict.copy()
        config_dict["imputation_strategy"] = "mode"
        config = PreprocessingConfig(**config_dict)
        preprocessor = DataPreprocessor(config)

        df_processed = preprocessor.handle_missing_values(sample_housing_data)

        assert len(df_processed) == len(sample_housing_data)
        assert df_processed["total_bedrooms"].isnull().sum() == 0

    def test_create_features_enabled(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test creating engineered features."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        df_processed, new_features = preprocessor.create_features(sample_housing_data)

        assert len(new_features) == 3
        assert "rooms_per_household" in new_features
        assert "bedrooms_per_room" in new_features
        assert "population_per_household" in new_features
        assert "rooms_per_household" in df_processed.columns
        assert "bedrooms_per_room" in df_processed.columns
        assert "population_per_household" in df_processed.columns

    def test_create_features_disabled(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test feature creation is skipped when disabled."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config_dict = preprocessing_config_dict.copy()
        config_dict["create_features"] = False
        config = PreprocessingConfig(**config_dict)
        preprocessor = DataPreprocessor(config)

        df_processed, new_features = preprocessor.create_features(sample_housing_data)

        assert len(new_features) == 0
        assert df_processed.equals(sample_housing_data)

    def test_create_features_missing_columns(
        self, preprocessing_config_dict, mock_gcs_client, monkeypatch
    ):
        """Test feature creation handles missing columns gracefully."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        # Create data with missing columns
        df_incomplete = pd.DataFrame({"longitude": [1, 2, 3], "latitude": [4, 5, 6]})

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        df_processed, new_features = preprocessor.create_features(df_incomplete)

        assert len(new_features) == 0

    def test_upload_to_gcs_success(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test successful upload to GCS."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        gcs_uri = preprocessor.upload_to_gcs(sample_housing_data)

        assert gcs_uri == f"gs://{config.bucket_name}/{config.gcs_output_path}"
        mock_gcs_client["blob"].upload_from_string.assert_called_once()
        assert mock_gcs_client["blob"].metadata is not None

    def test_upload_to_gcs_failure(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test upload to GCS fails with GoogleAPIError."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        mock_gcs_client["blob"].upload_from_string.side_effect = gcp_exceptions.GoogleAPIError(
            "Upload failed"
        )

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        with pytest.raises(RuntimeError, match="Error uploading to GCS"):
            preprocessor.upload_to_gcs(sample_housing_data)

    def test_run_success_with_median(
        self,
        preprocessing_config_dict,
        mock_gcs_client,
        sample_housing_data,
        monkeypatch,
        mock_mlflow,
    ):
        """Test successful full preprocessing run with median imputation."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        # Mock download
        csv_buffer = io.StringIO()
        sample_housing_data.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        mock_gcs_client["blob"].exists.return_value = True
        mock_gcs_client["blob"].download_as_bytes.return_value = csv_bytes

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        result = preprocessor.run()

        assert result.success is True
        assert result.error_message is None
        assert result.stats.input_rows == len(sample_housing_data)
        assert result.stats.output_rows == len(sample_housing_data)
        assert len(result.stats.new_features) == 3

    def test_run_success_with_drop(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test successful run with drop strategy."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        # Mock download
        csv_buffer = io.StringIO()
        sample_housing_data.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        mock_gcs_client["blob"].exists.return_value = True
        mock_gcs_client["blob"].download_as_bytes.return_value = csv_bytes

        config_dict = preprocessing_config_dict.copy()
        config_dict["imputation_strategy"] = "drop"
        config = PreprocessingConfig(**config_dict)
        preprocessor = DataPreprocessor(config)

        result = preprocessor.run()

        assert result.success is True
        assert result.stats.output_rows < result.stats.input_rows
        assert result.stats.rows_dropped > 0

    def test_run_failure(self, preprocessing_config_dict, mock_gcs_client, monkeypatch):
        """Test run returns error result on failure."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        # Mock download to fail
        mock_gcs_client["blob"].exists.return_value = True
        mock_gcs_client["blob"].download_as_bytes.side_effect = Exception("Download failed")

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        result = preprocessor.run()

        assert result.success is False
        assert result.error_message is not None

    def test_analyze_and_select_best_imputation(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test automatic imputation method selection."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        df_imputed = preprocessor.analyze_and_select_best_imputation(sample_housing_data)

        assert len(df_imputed) == len(sample_housing_data)
        assert df_imputed["total_bedrooms"].isnull().sum() == 0
        assert preprocessor.best_imputation_method != ""
        assert len(preprocessor.imputation_metrics) > 0

    def test_get_imputation_plots_with_analyzer(
        self, preprocessing_config_dict, mock_gcs_client, sample_housing_data, monkeypatch
    ):
        """Test generating imputation plots."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        # First run analysis to create analyzer
        preprocessor.analyze_and_select_best_imputation(sample_housing_data)

        plots = preprocessor.get_imputation_plots()

        assert "correlation_heatmap" in plots
        assert "imputation_comparison" in plots

    def test_get_imputation_plots_without_analyzer(
        self, preprocessing_config_dict, mock_gcs_client, monkeypatch
    ):
        """Test getting plots returns empty dict when no analyzer exists."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        plots = preprocessor.get_imputation_plots()

        assert plots == {}

    def test_create_wandb_metadata(self, preprocessing_config_dict, mock_gcs_client, monkeypatch):
        """Test creation of W&B metadata."""
        monkeypatch.setattr("preprocessor.storage.Client", lambda: mock_gcs_client["client"])

        config = PreprocessingConfig(**preprocessing_config_dict)
        preprocessor = DataPreprocessor(config)

        stats = PreprocessingStats(
            input_rows=100,
            input_columns=9,
            output_rows=80,
            output_columns=12,
            input_size_mb=0.5,
            output_size_mb=0.6,
            new_features=["rooms_per_household", "bedrooms_per_room", "population_per_household"],
        )

        result = PreprocessingResult(
            gcs_input_uri=f"gs://{config.bucket_name}/{config.gcs_input_path}",
            gcs_output_uri=f"gs://{config.bucket_name}/{config.gcs_output_path}",
            stats=stats,
            artifact_name=config.artifact_name,
            success=True,
        )

        metadata = preprocessor.create_wandb_metadata(result)

        assert metadata.input_artifact == config.input_artifact_name
        assert metadata.input_rows == 100
        assert metadata.output_rows == 80
        assert metadata.rows_dropped == 20
        assert metadata.columns_added == 3
        assert len(metadata.new_features) == 3
