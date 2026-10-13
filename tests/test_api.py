"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from src.model_utils import model_exists, DEFAULT_MODEL_PATH
from src.train import train


@pytest.fixture(scope="module", autouse=True)
def setup_api():
    """Ensure a trained model exists for API testing."""
    if not model_exists(DEFAULT_MODEL_PATH):
        train()


client = TestClient(app)


def test_root_endpoint():
    """Verify that GET / returns the running message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Customer Churn Prediction API is running"}


def test_health_endpoint():
    """Verify that GET /health reports healthy status and model_loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_valid_customer():
    """Verify that POST /predict returns valid prediction and probability."""
    payload = {
        "tenure": 12,
        "monthly_charges": 82.4,
        "total_charges": 988.8,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "churn" in data
    assert isinstance(data["churn"], bool)
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_invalid_customer():
    """Verify that invalid inputs (e.g. negative tenure) are rejected with HTTP 422."""
    invalid_payload = {
        "tenure": -5,
        "monthly_charges": 82.4,
        "total_charges": 988.8,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
