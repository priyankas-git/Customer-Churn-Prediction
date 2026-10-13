import os
import sys
from typing import Any, Dict

# Ensure project root is in sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.model_utils import load_model, model_exists


def predict_churn(customer_data: Dict[str, Any], model=None) -> Dict[str, Any]:
    """
    Generate churn prediction and probability for customer features.

    Args:
        customer_data: Dictionary containing customer feature values.
        model: Optional pre-loaded scikit-learn pipeline. If None, loaded from disk.

    Returns:
        Dict with keys: prediction (0 or 1), churn (bool), probability (float).
    """
    if model is None:
        model = load_model()

    # Convert single input dict to a DataFrame with identical schema
    df = pd.DataFrame([customer_data])
    raw_pred = model.predict(df)[0]
    pred = int(raw_pred)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df)[0]
        classes = list(model.classes_)
        if 1 in classes:
            churn_idx = classes.index(1)
            prob = float(probabilities[churn_idx])
        else:
            prob = float(probabilities[1]) if len(probabilities) > 1 else float(pred)
    else:
        prob = float(pred)

    return {
        "prediction": pred,
        "churn": bool(pred == 1),
        "probability": round(prob, 2),
    }


if __name__ == "__main__":
    example_customer = {
        "tenure": 12,
        "monthly_charges": 82.4,
        "total_charges": 988.8,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No",
    }
    if model_exists():
        result = predict_churn(example_customer)
        print("Example prediction result:", result)
    else:
        print("Model file does not exist yet. Please run src/train.py first.")
