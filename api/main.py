"""
MLOps Depression Prediction API
BCU CMP5366 – Bikash Kushwaha
"""

import os
import json
import joblib
import redis
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from sqlalchemy import create_engine, text

# ── Prometheus Metrics ────────────────────────────────────────────────────────
REQUEST_COUNT    = Counter("api_requests_total",    "Total API requests",          ["endpoint", "method", "status"])
PREDICTION_COUNT = Counter("predictions_total",     "Total predictions made",       ["result"])
LATENCY          = Histogram("request_latency_seconds", "Request latency",          ["endpoint"])
DRIFT_GAUGE      = Gauge("data_drift_score",        "Latest Evidently drift score")
MODEL_VERSION    = Gauge("model_version",           "Current model version number")

# ── App Initialisation ────────────────────────────────────────────────────────
app = FastAPI(
    title="Depression Risk Prediction API",
    description="MLOps pipeline API for predicting depression risk in working professionals. BCU CMP5366.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database & Redis ──────────────────────────────────────────────────────────
DB_URL = (
    f"mysql+mysqlclient://{os.getenv('DB_USER','mlops')}:{os.getenv('DB_PASSWORD','mlops123')}"
    f"@{os.getenv('DB_HOST','mariadb')}:{os.getenv('DB_PORT','3306')}/{os.getenv('DB_NAME','depression_db')}"
)
engine = create_engine(DB_URL, pool_pre_ping=True)

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
)

# ── Load Model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("/app/models/depression_model.pkl")
model = None

def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        MODEL_VERSION.set(1)
        print(f"[INFO] Model loaded from {MODEL_PATH}")
    else:
        print("[WARN] No trained model found. Run the training DAG first.")

load_model()

# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    age: int                        = Field(..., ge=18, le=70,  description="Age of the professional")
    work_hours_per_week: float      = Field(..., ge=0,  le=100, description="Average weekly working hours")
    years_experience: float         = Field(..., ge=0,  le=50,  description="Years of professional experience")
    job_satisfaction: int           = Field(..., ge=1,  le=10,  description="Job satisfaction score (1–10)")
    sleep_hours: float              = Field(..., ge=0,  le=12,  description="Average daily sleep hours")
    physical_activity_days: int     = Field(..., ge=0,  le=7,   description="Days per week with physical activity")
    social_interactions_per_week: int = Field(..., ge=0, le=50, description="Social interactions per week")
    has_mental_health_support: bool = Field(..., description="Access to mental health support at work")
    remote_work: bool               = Field(..., description="Works remotely")

    @validator("sleep_hours")
    def sleep_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Sleep hours cannot be negative")
        return v

class PredictionOutput(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    timestamp: str
    model_version: str
    cached: bool = False

class DriftReport(BaseModel):
    drift_detected: bool
    drift_score: float
    timestamp: str
    features_drifted: list[str]

class HealthResponse(BaseModel):
    status: str
    db: str
    redis: str
    model_loaded: bool
    timestamp: str

# ── Helper Functions ──────────────────────────────────────────────────────────
def input_to_dataframe(data: PredictionInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "age": data.age,
        "work_hours_per_week": data.work_hours_per_week,
        "years_experience": data.years_experience,
        "job_satisfaction": data.job_satisfaction,
        "sleep_hours": data.sleep_hours,
        "physical_activity_days": data.physical_activity_days,
        "social_interactions_per_week": data.social_interactions_per_week,
        "has_mental_health_support": int(data.has_mental_health_support),
        "remote_work": int(data.remote_work),
    }])

def risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    return "LOW"

def log_prediction_to_db(input_data: dict, result: str, prob: float):
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO predictions (input_data, prediction, probability, created_at)
                VALUES (:input_data, :prediction, :probability, NOW())
            """), {
                "input_data": json.dumps(input_data),
                "prediction": result,
                "probability": prob,
            })
            conn.commit()
    except Exception as e:
        print(f"[WARN] DB log failed: {e}")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    REQUEST_COUNT.labels(endpoint="/health", method="GET", status="200").inc()

    db_status = "ok"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {e}"

    redis_status = "ok"
    try:
        redis_client.ping()
    except Exception as e:
        redis_status = f"error: {e}"

    return HealthResponse(
        status="healthy" if db_status == "ok" and redis_status == "ok" else "degraded",
        db=db_status,
        redis=redis_status,
        model_loaded=model is not None,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(data: PredictionInput, background_tasks: BackgroundTasks):
    """
    Predict depression risk for a working professional.
    Results are cached in Redis for 10 minutes.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the training DAG first.")

    with LATENCY.labels(endpoint="/predict").time():
        # Check Redis cache
        cache_key = f"pred:{hash(str(data.dict()))}"
        cached = redis_client.get(cache_key)
        if cached:
            result = json.loads(cached)
            result["cached"] = True
            REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()
            return PredictionOutput(**result)

        df = input_to_dataframe(data)
        prob = float(model.predict_proba(df)[0][1])
        label = "Depressed" if prob >= 0.5 else "Not Depressed"

        PREDICTION_COUNT.labels(result=label).inc()

        output = {
            "prediction": label,
            "probability": round(prob, 4),
            "risk_level": risk_level(prob),
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": "1.0.0",
            "cached": False,
        }

        # Cache for 10 minutes
        redis_client.setex(cache_key, 600, json.dumps(output))

        # Async DB log
        background_tasks.add_task(log_prediction_to_db, data.dict(), label, prob)

    REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()
    return PredictionOutput(**output)


@app.get("/drift", response_model=DriftReport, tags=["Monitoring"])
def get_drift_report():
    """Return the latest Evidently AI drift report from Redis."""
    cached = redis_client.get("drift:latest")
    if not cached:
        raise HTTPException(status_code=404, detail="No drift report available. Run the monitoring DAG.")

    report = json.loads(cached)
    DRIFT_GAUGE.set(report.get("drift_score", 0))
    return DriftReport(**report)


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/reload", tags=["System"])
def reload_model():
    """Hot-reload the model from disk (called by Airflow after retraining)."""
    load_model()
    return {"status": "reloaded", "model_path": str(MODEL_PATH), "timestamp": datetime.utcnow().isoformat()}


@app.get("/predictions/history", tags=["Prediction"])
def prediction_history(limit: int = 50):
    """Fetch recent predictions from MariaDB."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT id, prediction, probability, created_at FROM predictions ORDER BY created_at DESC LIMIT :lim"
            ), {"lim": limit}).fetchall()
        return [{"id": r[0], "prediction": r[1], "probability": r[2], "created_at": str(r[3])} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
