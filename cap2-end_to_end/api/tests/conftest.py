"""
Pytest configuration and fixtures for API tests.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add parent directory to path for imports
api_dir = Path(__file__).parent.parent
sys.path.insert(0, str(api_dir))

from app.core.model_loader import ModelLoader
from app.main import app
from app.routers import predict


class MockModel:
    """Mock ML model for testing."""

    def predict(self, X):
        """Return mock predictions."""
        return np.array([250000.0 + i * 1000 for i in range(len(X))])


@pytest.fixture
def mock_model_loader():
    """Create a mock model loader with loaded model."""
    loader = Mock(spec=ModelLoader)
    loader.is_loaded = True
    loader.model_version = "test_model_v1"
    loader._model = MockModel()

    def mock_predict(features):
        return loader._model.predict(features)

    loader.predict = mock_predict
    return loader


@pytest.fixture
def client(mock_model_loader):
    """Create test client with mocked model."""
    # Set mock model loader in router
    predict.set_model_loader(mock_model_loader)

    # Set in app state
    app.state.model_loader = mock_model_loader

    # Create test client
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_housing_data():
    """Sample housing data for testing."""
    return {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
    }


@pytest.fixture
def sample_prediction_request(sample_housing_data):
    """Sample prediction request."""
    return {"instances": [sample_housing_data]}


@pytest.fixture
def sample_batch_prediction_request(sample_housing_data):
    """Sample batch prediction request."""
    return {
        "instances": [
            sample_housing_data,
            {**sample_housing_data, "median_income": 5.0},
            {**sample_housing_data, "housing_median_age": 20.0},
        ]
    }
