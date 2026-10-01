# House Price Prediction - End-to-End MLOps

FastAPI + MLflow (tracking + model registry) + Docker + GitHub Actions + Docker Hub + AWS EC2

## Flow
git push -> GitHub Actions -> image Docker Hub pe (anujkrpathak/mlops)
EC2: docker compose -> MLflow (5000) + App (8000); trainer job model register karta hai (alias: champion)

## EC2 pe commands
```bash
docker compose pull
docker compose up -d mlflow app
docker compose run --rm trainer      # train + register champion
docker compose restart app           # naya model load karne ke liye
```
App: http://EC2-IP:8000   |   MLflow: http://EC2-IP:5000

## Local test
```bash
docker compose up -d --build mlflow app
docker compose run --rm trainer
```
http://localhost:8000
