"""
Tests for API endpoints.
"""

from fastapi import status


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs_url" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["model_loaded"] is True

    def test_health_check_structure(self, client):
        """Test health check response structure."""
        response = client.get("/health")
        data = response.json()

        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["version"], str)


class TestPredictionEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_single_instance(self, client, sample_prediction_request):
        """Test prediction with single instance."""
        response = client.post("/api/v1/predict", json=sample_prediction_request)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "predictions" in data
        assert "model_version" in data
        assert len(data["predictions"]) == 1
        assert "predicted_price" in data["predictions"][0]

    def test_predict_batch(self, client, sample_batch_prediction_request):
        """Test batch prediction."""
        response = client.post("/api/v1/predict", json=sample_batch_prediction_request)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert len(data["predictions"]) == 3
        for pred in data["predictions"]:
            assert "predicted_price" in pred
            assert isinstance(pred["predicted_price"], (int, float))

    def test_predict_invalid_ocean_proximity(self, client, sample_housing_data):
        """Test prediction with invalid ocean proximity."""
        invalid_data = {**sample_housing_data, "ocean_proximity": "INVALID"}
        request = {"instances": [invalid_data]}

        response = client.post("/api/v1/predict", json=request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_missing_field(self, client, sample_housing_data):
        """Test prediction with missing required field."""
        incomplete_data = {**sample_housing_data}
        del incomplete_data["longitude"]
        request = {"instances": [incomplete_data]}

        response = client.post("/api/v1/predict", json=request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_negative_values(self, client, sample_housing_data):
        """Test prediction with invalid negative values."""
        invalid_data = {**sample_housing_data, "total_rooms": -100}
        request = {"instances": [invalid_data]}

        response = client.post("/api/v1/predict", json=request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_response_structure(self, client, sample_prediction_request):
        """Test prediction response has correct structure."""
        response = client.post("/api/v1/predict", json=sample_prediction_request)

        data = response.json()
        assert isinstance(data["predictions"], list)
        assert isinstance(data["model_version"], str)

        prediction = data["predictions"][0]
        assert isinstance(prediction["predicted_price"], (int, float))
        assert prediction["predicted_price"] > 0


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert data["model_loaded"] is True
        assert data["model_version"] == "test_model_v1"


class TestValidation:
    """Tests for input validation."""

    def test_empty_instances_list(self, client):
        """Test with empty instances list."""
        request = {"instances": []}

        response = client.post("/api/v1/predict", json=request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_ocean_proximity_case_insensitive(self, client, sample_housing_data):
        """Test ocean proximity accepts different cases."""
        data = {**sample_housing_data, "ocean_proximity": "near bay"}
        request = {"instances": [data]}

        response = client.post("/api/v1/predict", json=request)

        assert response.status_code == status.HTTP_200_OK

    def test_latitude_longitude_bounds(self, client, sample_housing_data):
        """Test latitude and longitude validation."""
        # Invalid latitude
        invalid_lat = {**sample_housing_data, "latitude": 91.0}
        response = client.post("/api/v1/predict", json={"instances": [invalid_lat]})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Invalid longitude
        invalid_lon = {**sample_housing_data, "longitude": 181.0}
        response = client.post("/api/v1/predict", json={"instances": [invalid_lon]})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
