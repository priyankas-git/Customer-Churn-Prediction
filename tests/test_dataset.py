"""Tests for customer churn dataset validation."""

import os
import pandas as pd
import pytest

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "churn.csv")
REQUIRED_COLUMNS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "internet_service",
    "tech_support",
    "churn",
]


def test_dataset_exists():
    """Verify that the dataset file exists."""
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"


def test_dataset_not_empty():
    """Verify that the dataset contains rows."""
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Dataset is empty"
    assert len(df) > 0, "Dataset contains 0 rows"


def test_dataset_required_columns():
    """Verify that all required columns are present in the dataset."""
    df = pd.read_csv(DATA_PATH)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Required column '{col}' is missing from dataset"


def test_churn_values():
    """Verify that the target column churn contains only 0 and 1."""
    df = pd.read_csv(DATA_PATH)
    unique_values = set(df["churn"].unique())
    assert unique_values.issubset({0, 1}), f"Unexpected values in churn column: {unique_values}"
