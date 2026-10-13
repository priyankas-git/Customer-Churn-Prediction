"""Pydantic schemas for request validation and response models."""

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    tenure: float = Field(..., ge=0, description="Customer tenure in months")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    total_charges: float = Field(..., ge=0, description="Total charges")
    contract_type: str = Field(..., description="Contract type: Month-to-month, One year, or Two year")
    internet_service: str = Field(..., description="Internet service: DSL or Fiber optic")
    tech_support: str = Field(..., description="Tech support: Yes or No")


class PredictionResponse(BaseModel):
    prediction: int
    churn: bool
    probability: float
