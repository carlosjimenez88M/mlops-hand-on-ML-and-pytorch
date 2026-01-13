"""
Common fixtures and configuration for pytest tests
Author: Carlos Daniel Jiménez
Date: 2025-01-13
"""

import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, Mock

import pytest
import pandas as pd
import numpy as np
from google.cloud import storage

# Add src directories to path - order matters!
project_root = Path(__file__).parent.parent
download_data_path = project_root / "src" / "data" / "01_download_data"
preprocessing_path = project_root / "src" / "data" / "02_preprocessing_and_imputation"

# Insert at position 0 to ensure our modules are found first
if str(download_data_path) not in sys.path:
    sys.path.insert(0, str(download_data_path))
if str(preprocessing_path) not in sys.path:
    sys.path.insert(0, str(preprocessing_path))


@pytest.fixture
def sample_housing_data():
    """Creates sample housing data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        'longitude': np.random.uniform(-124, -114, n_samples),
        'latitude': np.random.uniform(32, 42, n_samples),
        'housing_median_age': np.random.randint(1, 53, n_samples),
        'total_rooms': np.random.randint(500, 5000, n_samples),
        'total_bedrooms': np.random.randint(100, 1000, n_samples),
        'population': np.random.randint(500, 3000, n_samples),
        'households': np.random.randint(100, 1000, n_samples),
        'median_income': np.random.uniform(0.5, 15, n_samples),
        'median_house_value': np.random.uniform(50000, 500000, n_samples)
    }

    df = pd.DataFrame(data)

    # Add some missing values to total_bedrooms
    missing_indices = np.random.choice(n_samples, size=20, replace=False)
    df.loc[missing_indices, 'total_bedrooms'] = np.nan

    return df


@pytest.fixture
def sample_housing_data_csv(sample_housing_data, tmp_path):
    """Saves sample housing data to a CSV file."""
    csv_path = tmp_path / "housing.csv"
    sample_housing_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_gcs_client():
    """Creates a mock GCS client."""
    mock_client = MagicMock(spec=storage.Client)
    mock_bucket = MagicMock(spec=storage.Bucket)
    mock_blob = MagicMock(spec=storage.Blob)

    # Configure bucket to return True on exists()
    mock_bucket.exists.return_value = True
    mock_bucket.blob.return_value = mock_blob

    # Configure client to return bucket
    mock_client.bucket.return_value = mock_bucket

    return {
        'client': mock_client,
        'bucket': mock_bucket,
        'blob': mock_blob
    }


@pytest.fixture
def mock_requests_response():
    """Creates a mock requests response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {'content-length': '1024'}
    mock_response.raise_for_status = Mock()
    return mock_response


@pytest.fixture
def download_config_dict():
    """Sample configuration for DataDownloader."""
    return {
        'file_url': 'https://example.com/housing.csv',
        'artifact_name': 'housing_data_raw',
        'artifact_type': 'raw_data',
        'artifact_description': 'California housing dataset',
        'gcs_output_path': 'data/01-raw/housing.csv',
        'bucket_name': 'test-bucket',
        'wandb_project': 'housing-mlops-gcp'
    }


@pytest.fixture
def preprocessing_config_dict():
    """Sample configuration for DataPreprocessor."""
    return {
        'input_artifact_name': 'housing_data_raw:latest',
        'gcs_input_path': 'data/01-raw/housing.csv',
        'gcs_output_path': 'data/02-processed/housing_processed.csv',
        'artifact_name': 'housing_data_processed',
        'artifact_type': 'processed_data',
        'artifact_description': 'Processed housing data',
        'bucket_name': 'test-bucket',
        'wandb_project': 'housing-mlops-gcp',
        'imputation_strategy': 'median',
        'create_features': True
    }


@pytest.fixture
def mock_mlflow(monkeypatch):
    """Mocks MLflow functions."""
    mock_log_metric = Mock()
    mock_log_param = Mock()
    mock_log_artifact = Mock()

    monkeypatch.setattr('mlflow.log_metric', mock_log_metric)
    monkeypatch.setattr('mlflow.log_param', mock_log_param)
    monkeypatch.setattr('mlflow.log_artifact', mock_log_artifact)

    return {
        'log_metric': mock_log_metric,
        'log_param': mock_log_param,
        'log_artifact': mock_log_artifact
    }
