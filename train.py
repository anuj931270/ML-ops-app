"""House price model: training + MLflow tracking + model registry.

Chalane ka tareeka:
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python train.py

Agar data/houses.csv nahi mili to synthetic (practice) data khud ban jata hai.
Apna real data use karna ho to CSV mein ye columns rakho:
    sqft, bedrooms, bathrooms, age_years, garage, location_score, price
"""
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DATA_PATH = Path(os.getenv("DATA_PATH", "data/houses.csv"))

EXPERIMENT_NAME = "house-price-prediction"
MODEL_NAME = "house-price-predictor"   # serve.py isi naam se model load karta hai
MODEL_ALIAS = "champion"

FEATURES = ["sqft", "bedrooms", "bathrooms", "age_years", "garage", "location_score"]
TARGET = "price"

N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "200"))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "12"))


def generate_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Practice ke liye synthetic house data."""
    rng = np.random.default_rng(seed)
    sqft = rng.integers(500, 4500, n)
    bedrooms = np.clip(np.round(sqft / 900 + rng.integers(-1, 2, n)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 5).astype(int)
    age_years = rng.integers(0, 60, n)
    garage = rng.integers(0, 4, n)
    location_score = rng.integers(1, 11, n)

    price = (
        sqft * 120
        + bedrooms * 8000
        + bathrooms * 12000
        - age_years * 900
        + garage * 9000
        + location_score * 18000
        + rng.normal(0, 15000, n)
    )
    return pd.DataFrame(
        {
            "sqft": sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "garage": garage,
            "location_score": location_score,
            "price": np.clip(price, 20000, None).round(2),
        }
    )


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        print(f"Data load ho raha hai: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print("CSV nahi mili, synthetic data ban raha hai...")
    df = generate_data()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def main():
    df = load_data().dropna()
    print(f"Data shape: {df.shape}")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {TRACKING_URI}")

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("rows", len(df))
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Model log karo aur registry mein register karo
        # MLflow sklearn models ko skops format mein save karta hai; RandomForest ke
        # Tree type ko trusted batana padta hai (tumhara apna trained model hai, safe hai).
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

    # Latest version ko 'champion' alias do
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, str(latest))
    print(f"Alias '{MODEL_ALIAS}' -> {MODEL_NAME} version {latest}")


if __name__ == "__main__":
    main()
