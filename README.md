# Customer Churn Prediction API

A production-ready ML API that predicts customer churn probability using RandomForestClassifier.

## Business Problem

Identify at-risk customers before they leave — missing a churner costs more than a false alarm.

## Model Performance

| Metric | Default (0.5) | Tuned (0.3) |
|--------|--------------|-------------|
| Recall | 0.44 | **0.76** |
| Precision | 0.64 | 0.54 |
| ROC-AUC | 0.84 | — |

## Tech Stack

- Python, scikit-learn, RandomForestClassifier
- FastAPI, Pydantic
- Docker, Azure Container Apps
- GitHub Actions CI/CD

## Live API

https://churn-v3-app.icyriver-c5d86bfa.eastus.azurecontainerapps.io/docs

## Endpoints

- GET /health — API status
- GET /metrics — Model KPIs
- POST /predict — Churn prediction

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
