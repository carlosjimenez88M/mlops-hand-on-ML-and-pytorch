"""API smoke tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app.main import app
from api.app.routers.predict import set_model_service


class DummyService:
    expected_input_dim = 4

    def predict(self, instances):
        return {
            "predictions": [1 for _ in instances],
            "probabilities": [[0.1, 0.9] for _ in instances],
            "model_name": "dummy",
            "model_version": "1",
        }


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200


def test_prediction_endpoint():
    set_model_service(DummyService())
    client = TestClient(app)
    response = client.post("/predict", json={"instances": [[0.0, 1.0, 2.0, 3.0]]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["predictions"] == [1]
