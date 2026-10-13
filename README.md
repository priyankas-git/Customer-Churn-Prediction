# Customer Churn Prediction System

A beginner-friendly, end-to-end Machine Learning Operations (MLOps) project for customer churn prediction. This application loads customer data, trains a Random Forest classification pipeline, tracks metrics with MLflow, enforces model validation thresholds, exposes a FastAPI backend, and includes a lightweight web frontend.

---

## 1. Project Overview

Customer churn occurs when customers stop doing business with a company. Predicting customer churn allows businesses to proactively identify at-risk customers and implement retention strategies.

This project demonstrates a minimal yet complete ML lifecycle:
- Data loading and preprocessing with `scikit-learn` `ColumnTransformer` and `Pipeline`
- Model training with `RandomForestClassifier`
- Experiment tracking with `MLflow` (logging parameters, metrics, and models)
- Quality gate validation based on an F1-score threshold (`F1 >= 0.80`)
- Model persistence and serving with `FastAPI`
- Clean, minimal web interface with HTML, CSS, and Vanilla JavaScript
- Automated test suite with `pytest`

---

## 2. Technologies Used

- **Language:** Python 3.12+
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **Machine Learning:** pandas, scikit-learn (RandomForestClassifier, ColumnTransformer, Pipeline)
- **Experiment Tracking:** MLflow (SQLite backend)
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Testing:** pytest, httpx

---

## 3. Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── churn.csv              # Customer dataset
│
├── src/
│   ├── __init__.py
│   ├── train.py               # ML training, evaluation, and MLflow logging
│   ├── predict.py             # Single-instance inference utility
│   └── model_utils.py         # Model saving and loading helpers
│
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI endpoints and static file serving
│   └── schemas.py             # Pydantic input/output schemas
│
├── frontend/
│   ├── index.html             # Customer churn input form
│   ├── style.css              # Clean, modern styles
│   └── script.js              # Fetch API request & UI interaction
│
├── models/
│   └── churn_model.pkl        # Saved trained model pipeline
│
├── tests/
│   ├── test_dataset.py        # Dataset structure and value tests
│   ├── test_model.py          # Model existence and inference tests
│   └── test_api.py            # FastAPI endpoint tests
│
├── requirements.txt           # Project dependencies
├── README.md                  # Documentation and instructions
└── .gitignore                 # Git ignore rules
```

---

## 4. How to Create Virtual Environment

Open PowerShell or terminal in the project directory:

```powershell
python -m venv venv
```

Activate the virtual environment:

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

---

## 5. How to Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 6. How to Train the Model

Run the training pipeline:

```powershell
python src/train.py
```

The script will:
1. Load `data/churn.csv` and display shape and class distribution.
2. Train a `RandomForestClassifier` inside an end-to-end preprocessing `Pipeline`.
3. Evaluate and display **Accuracy**, **Precision**, **Recall**, and **F1 Score**.
4. Log parameters and metrics into MLflow.
5. Check if `F1 >= 0.80`. If passed, saves `models/churn_model.pkl`.

---

## 7. How to Start MLflow

To review logged runs, metrics, and parameters:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then visit:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 8. How to Start FastAPI

Start the backend server with Uvicorn:

```powershell
uvicorn app.main:app --reload
```

---

## 9. How to Use the Frontend

1. Ensure the FastAPI server is running (`uvicorn app.main:app --reload`).
2. Open your web browser and navigate to:
   [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
3. Fill in the customer details:
   - **Tenure** (e.g., `12` months)
   - **Monthly Charges** (e.g., `82.40`)
   - **Total Charges** (e.g., `988.80`)
   - **Contract Type** (`Month-to-month`, `One year`, or `Two year`)
   - **Internet Service** (`DSL` or `Fiber optic`)
   - **Tech Support** (`Yes` or `No`)
4. Click **Predict Churn**.
5. The page displays:
   - Prediction: **Likely to Churn** or **Likely to Stay**
   - Churn Probability: `XX%`

---

## 10. API Endpoints

Interactive Swagger UI documentation is available at:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### `GET /`
- Returns HTML frontend when visited in a browser.
- Returns status JSON for API clients:
  ```json
  {
    "message": "Customer Churn Prediction API is running"
  }
  ```

### `GET /health`
Returns health and model status:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`
Generates a churn prediction for customer data.

**Request Body:**
```json
{
  "tenure": 12,
  "monthly_charges": 82.4,
  "total_charges": 988.8,
  "contract_type": "Month-to-month",
  "internet_service": "Fiber optic",
  "tech_support": "No"
}
```

**Response Body:**
```json
{
  "prediction": 1,
  "churn": true,
  "probability": 0.82
}
```

---

## 11. How to Run Tests

Run automated tests using pytest:

```powershell
pytest
```

To run with verbose output:

```powershell
pytest -v
```
