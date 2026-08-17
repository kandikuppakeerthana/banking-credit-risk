from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os


# ==========================================
# FastAPI Application
# ==========================================

app = FastAPI(
    title="Banking Credit Risk API",
    description="API for credit default risk prediction",
    version="1.0.0"
)


# ==========================================
# Locate project files
# ==========================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "final_credit_risk_model.pkl"
)

FEATURE_PATH = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "final_features.pkl"
)


# ==========================================
# Load trained model and features
# ==========================================

model = joblib.load(MODEL_PATH)
selected_features = joblib.load(FEATURE_PATH)


print("Model loaded successfully!")
print("Number of features:", len(selected_features))


# ==========================================
# Input data structure
# ==========================================

class CustomerData(BaseModel):

    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int

    PAY_SEPT: int
    PAY_AUG: int
    PAY_JUL: int
    PAY_JUN: int
    PAY_MAY: int
    PAY_APR: int

    BILL_AMT_SEPT: float
    BILL_AMT_AUG: float
    BILL_AMT_JUL: float
    BILL_AMT_JUN: float
    BILL_AMT_MAY: float
    BILL_AMT_APR: float

    PAY_AMT_SEPT: float
    PAY_AMT_AUG: float
    PAY_AMT_JUL: float
    PAY_AMT_JUN: float
    PAY_AMT_MAY: float
    PAY_AMT_APR: float


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health_check():

    return {
        "message": "API is running"
    }


# ==========================================
# Prediction Endpoint
# ==========================================

@app.post("/predict")
def predict_credit_risk(customer: CustomerData):

    # Convert input into dictionary
    customer_data = customer.model_dump()

    # Convert dictionary to DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Keep exact 23 features
    customer_df = customer_df[selected_features]

    # Predict probability
    probability = model.predict_proba(customer_df)[0][1]

    # Final threshold from Phase 7
    FINAL_THRESHOLD = 0.25

    # Risk classification
    if probability >= FINAL_THRESHOLD:
        risk_category = "High Risk"
        banking_decision = "Further Review Required"
    else:
        risk_category = "Low Risk"
        banking_decision = "Standard Evaluation"

    return {
        "default_probability": round(float(probability), 6),
        "risk_category": risk_category,
        "banking_decision": banking_decision
    }