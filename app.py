from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train_model import prepare_customer_record

app = FastAPI(title='Customer Churn Prediction API')

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'model' / 'churn_model.pkl'
model_bundle = joblib.load(MODEL_PATH)
model_pipeline = model_bundle['pipeline']
feature_columns = model_bundle['feature_columns']


class CustomerInput(BaseModel):
    customerID: str = Field(default='unknown')
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get('/')
def root():
    return {'message': 'Customer Churn Prediction API is running.'}


@app.post('/predict')
def predict_churn(payload: CustomerInput):
    try:
        record = prepare_customer_record(payload.model_dump())
        model_input = {key: record.get(key, None) for key in feature_columns if key != 'Churn'}
        model_frame = pd.DataFrame([model_input], columns=feature_columns)
        prediction = model_pipeline.predict(model_frame)[0]
        probability = model_pipeline.predict_proba(model_frame)[0]
        churn_probability = float(probability[1]) if len(probability) > 1 else float(probability[0])
        return {
            'prediction': 'Yes' if prediction == 1 else 'No',
            'churn_probability': round(churn_probability, 4),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid input: {str(exc)}')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
