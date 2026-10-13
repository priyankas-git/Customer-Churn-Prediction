"""Tests for trained churn model pipeline."""

import os
import pandas as pd
import pytest
from src.model_utils import load_model, model_exists, DEFAULT_MODEL_PATH
from src.train import train


@pytest.fixture(scope="module", autouse=True)
def ensure_model_trained():
    """Ensure a trained model exists before running model tests."""
    if not model_exists(DEFAULT_MODEL_PATH):
        train()


def test_model_file_exists():
    """Verify that the model file exists after training."""
    assert model_exists(DEFAULT_MODEL_PATH), f"Model file not found at {DEFAULT_MODEL_PATH}"


def test_model_can_be_loaded():
    """Verify that the saved model pipeline can be loaded successfully."""
    model = load_model(DEFAULT_MODEL_PATH)
    assert model is not None, "Loaded model is None"
    assert hasattr(model, "predict"), "Loaded object does not implement predict"


def test_model_prediction_and_range():
    """Verify that the model can make predictions and the result is 0 or 1."""
    model = load_model(DEFAULT_MODEL_PATH)
    sample_data = pd.DataFrame([{
        "tenure": 12,
        "monthly_charges": 82.4,
        "total_charges": 988.8,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No",
    }])

    pred = model.predict(sample_data)[0]
    assert int(pred) in [0, 1], f"Expected prediction to be 0 or 1, got {pred}"

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample_data)[0]
        assert len(probs) == 2, f"Expected 2 class probabilities, got {len(probs)}"
        assert 0.0 <= probs[1] <= 1.0, f"Churn probability out of bounds: {probs[1]}"
