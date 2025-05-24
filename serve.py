# serve.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

import config

app = FastAPI(title="Telco Churn Prediction API")

# Define request schema
class CustomerData(BaseModel):
    # include known feature names, e.g.:
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
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Load model once at startup
@app.on_event("startup")
def load_model():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    try:
        app.state.model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}/latest")
        print("Model loaded from MLflow registry.")
    except Exception:
        import joblib
        app.state.model = joblib.load(config.MODEL_SAVE_PATH)
        print(f"Model loaded from local path: {config.MODEL_SAVE_PATH}")

@app.post("/predict")
def predict(data: CustomerData):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])
        # Preprocess similar to training
        from data import preprocess_data
        df_proc = preprocess_data(df)
        # Drop target if present
        if config.TARGET_COL in df_proc.columns:
            df_proc = df_proc.drop(config.TARGET_COL, axis=1)

        model = app.state.model
        pred = model.predict(df_proc)[0]
        prob = model.predict_proba(df_proc)[0, 1] if hasattr(model, "predict_proba") else None

        return {"prediction": int(pred), "churn_probability": float(prob) if prob is not None else None}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)