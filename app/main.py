"""FastAPI application for Customer Churn Prediction."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import CustomerInput, PredictionResponse
from src.model_utils import load_model, model_exists
from src.predict import predict_churn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# In-memory model cache
loaded_model = None


def get_or_load_model():
    """Retrieve cached model or load from disk if available."""
    global loaded_model
    if loaded_model is None and model_exists():
        loaded_model = load_model()
    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to load model on startup."""
    global loaded_model
    if model_exists():
        try:
            loaded_model = load_model()
        except Exception:
            loaded_model = None
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for API requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend directory if it exists
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def read_root(request: Request):
    """
    Root endpoint.
    Returns frontend HTML when requested by a browser (Accept: text/html),
    or the API status JSON message for API clients.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept and not accept.startswith("*/*"):
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
    return {"message": "Customer Churn Prediction API is running"}


@app.get("/app")
def read_frontend():
    """Direct route to access the frontend web interface."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend index.html not found")


@app.get("/style.css")
def read_style():
    """Serve style.css directly if requested from root."""
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"))


@app.get("/script.js")
def read_script():
    """Serve script.js directly if requested from root."""
    return FileResponse(os.path.join(FRONTEND_DIR, "script.js"))


@app.get("/health")
def health_check():
    """Health check endpoint indicating API and model status."""
    model = get_or_load_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    """
    Generate churn prediction and churn probability.
    Validates input using Pydantic, formats into pandas DataFrame,
    and runs inference through the trained pipeline.
    """
    model = get_or_load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Please train the model first.",
        )

    try:
        result = predict_churn(customer.model_dump(), model=model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
