import os
import sys

# Ensure project root is in sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

from src.model_utils import save_model, DEFAULT_MODEL_PATH

F1_THRESHOLD = 0.80


def train():
    # Base path resolution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "churn.csv")
    if os.path.exists("data/churn.csv"):
        data_path = "data/churn.csv"

    # Step 1 — Load dataset
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully")
    print(f"Dataset shape: {df.shape}")
    print("Churn class counts:")
    print(df["churn"].value_counts())

    # Step 2 — Separate features and target
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Step 3 — Define feature types
    categorical_features = ["contract_type", "internet_service", "tech_support"]
    numerical_features = ["tenure", "monthly_charges", "total_charges"]

    # Step 4 — Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    # Step 5 — Model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    # Step 6 — Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    # Step 7 — Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train model pipeline
    pipeline.fit(X_train, y_train)

    # Step 8 — Model Evaluation
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nModel Training Completed!\n")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")

    # MLflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Customer Churn Prediction")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model", "Random Forest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("test_size", 0.2)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log the trained sklearn pipeline model with safe trusted types
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Step 9 — Model Validation
        if f1 >= F1_THRESHOLD:
            print("VALIDATION PASSED")
            print("Model is approved for registration.")
            mlflow.set_tag("validation_status", "PASSED")
            save_model(pipeline, DEFAULT_MODEL_PATH)
            print(f"Model saved to {DEFAULT_MODEL_PATH}")
        else:
            print("VALIDATION FAILED")
            print("Model is rejected.")
            mlflow.set_tag("validation_status", "FAILED")

    return pipeline, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}


if __name__ == "__main__":
    train()
