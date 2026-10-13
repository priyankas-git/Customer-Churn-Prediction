"""Model utility functions for saving, loading, and checking model artifacts."""

import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")


def save_model(model, filepath: str = DEFAULT_MODEL_PATH) -> None:
    """Save the trained model pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str = DEFAULT_MODEL_PATH):
    """Load the trained model pipeline from disk."""
    if not model_exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)


def model_exists(filepath: str = DEFAULT_MODEL_PATH) -> bool:
    """Check if the trained model file exists."""
    return os.path.exists(filepath)
