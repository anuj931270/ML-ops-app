"""FastAPI service: MLflow registry se 'champion' model load karke predictions deta hai."""
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
TRACKING_URI = os.getenv("MLFLOW_TRACKING", "http://127.0.0.1:5000")
MODEL_URI = "models:/house-price-predictor@champion"
FEATURES = ["sqft", "bedrooms", "bathrooms", "age_years", "garage", "location_score"]

mlflow.set_tracking_uri(TRACKING_URI)
_model = None


def get_model():
    """Model pehli baar chahiye tab load hota hai. Load fail ho to app crash nahi hoti."""
    global _model
    if _model is None:
        _model = mlflow.sklearn.load_model(MODEL_URI)
    return _model


try:
    get_model()
    print(f"Model loaded: {MODEL_URI}")
except Exception as exc:  # training abhi nahi hui ho sakti
    print(f"Model abhi load nahi hua: {exc}")

app = FastAPI(title="House Price Predictor")


class HouseFeatures(BaseModel):
    sqft: float = Field(..., gt=0, le=20000)
    bedrooms: int = Field(..., gt=0, le=20)
    bathrooms: int = Field(..., gt=0, le=20)
    age_years: int = Field(..., ge=0, le=200)
    garage: int = Field(..., ge=0, le=10)
    location_score: int = Field(..., ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model is not None, "model": MODEL_URI}


@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        model = get_model()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Model abhi available nahi hai. Pehle training chalao (docker compose run --rm trainer).",
        )
    input_df = pd.DataFrame([features.model_dump()], columns=FEATURES)
    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(float(prediction), 2)}


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def frontend():
    return FileResponse(BASE_DIR / "static" / "index.html")
