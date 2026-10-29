import joblib
import pandas as pd 
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("models/fraud_model.joblib")
scaler = joblib.load("models/scaler.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")

app = FastAPI(title="Credit Card Fraud Detection API")


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/")
def home():
   return {"message": "Fraud detection is running"}



@app.post("/predict")
def predict(transaction : Transaction):
    data = pd.DataFrame([transaction.model_dump()])
    data = data[feature_columns]
    data[["Time","Amount"]] = scaler.transform(data[["Time","Amount"]])


    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return{
        "prediction": prediction,
        "label": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability":  round(probability,4),
    }
