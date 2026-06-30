"""
DAG: depression_model_training
Purpose: Train Random Forest classifier for depression risk prediction
Schedule: Weekly on Monday
BCU CMP5366 – Bikash Kushwaha
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import json
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "bikash",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

dag = DAG(
    dag_id="depression_model_training",
    default_args=default_args,
    description="Train and evaluate depression prediction model",
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "training", "cmp5366"],
)

DATA_PATH   = "/opt/airflow/data"
MODEL_PATH  = "/opt/airflow/models"
CLEAN_FILE  = f"{DATA_PATH}/clean_depression_data.csv"
MODEL_FILE  = f"{MODEL_PATH}/depression_model.pkl"
METRICS_FILE = f"{MODEL_PATH}/metrics.json"
THRESHOLD   = 0.75  # Minimum acceptable accuracy


def prepare_features(**context):
    """Load cleaned data and split into train/test sets."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(CLEAN_FILE)
    feature_cols = [
        "age", "work_hours_per_week", "years_experience", "job_satisfaction",
        "sleep_hours", "physical_activity_days", "social_interactions_per_week",
        "has_mental_health_support", "remote_work",
        "work_life_balance", "activity_sleep_ratio",
    ]
    # Only use columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["depressed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(DATA_PATH, exist_ok=True)
    X_train.to_csv(f"{DATA_PATH}/X_train.csv", index=False)
    X_test.to_csv(f"{DATA_PATH}/X_test.csv",  index=False)
    y_train.to_csv(f"{DATA_PATH}/y_train.csv", index=False)
    y_test.to_csv(f"{DATA_PATH}/y_test.csv",   index=False)

    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")
    context["ti"].xcom_push(key="feature_cols", value=feature_cols)


def train_model(**context):
    """Train Random Forest with cross-validation."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_train = pd.read_csv(f"{DATA_PATH}/X_train.csv")
    y_train = pd.read_csv(f"{DATA_PATH}/y_train.csv").squeeze()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"[INFO] CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(X_train, y_train)

    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"[INFO] Model saved → {MODEL_FILE}")

    context["ti"].xcom_push(key="cv_mean", value=float(cv_scores.mean()))


def evaluate_model(**context):
    """Evaluate model on held-out test set and save metrics."""
    from sklearn.metrics import (
        accuracy_score, classification_report,
        roc_auc_score, f1_score, confusion_matrix,
    )

    pipeline = joblib.load(MODEL_FILE)
    X_test = pd.read_csv(f"{DATA_PATH}/X_test.csv")
    y_test = pd.read_csv(f"{DATA_PATH}/y_test.csv").squeeze()

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "evaluated_at": datetime.utcnow().isoformat(),
        "model_path": MODEL_FILE,
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Accuracy: {metrics['accuracy']}, AUC: {metrics['roc_auc']}")
    context["ti"].xcom_push(key="accuracy", value=metrics["accuracy"])


def check_model_quality(**context):
    """Branch: promote model if accuracy >= threshold, else fail."""
    accuracy = context["ti"].xcom_pull(task_ids="evaluate_model", key="accuracy")
    print(f"[INFO] Model accuracy: {accuracy}, threshold: {THRESHOLD}")
    if accuracy >= THRESHOLD:
        return "promote_model"
    return "reject_model"


def promote_model(**context):
    """Promote model and notify FastAPI to reload."""
    import shutil
    import httpx

    # Copy to /app/models for FastAPI
    shutil.copy(MODEL_FILE, "/opt/airflow/models/depression_model.pkl")
    print("[INFO] Model promoted.")

    # Hot-reload FastAPI
    try:
        r = httpx.get("http://fastapi:8000/model/reload", timeout=10)
        print(f"[INFO] FastAPI reload response: {r.status_code}")
    except Exception as e:
        print(f"[WARN] FastAPI reload failed: {e}")


# ── Tasks ──────────────────────────────────────────────────────────────────────
t_prepare = PythonOperator(task_id="prepare_features",  python_callable=prepare_features, dag=dag)
t_train   = PythonOperator(task_id="train_model",       python_callable=train_model,      dag=dag)
t_eval    = PythonOperator(task_id="evaluate_model",    python_callable=evaluate_model,   dag=dag)

t_branch  = BranchPythonOperator(task_id="check_model_quality", python_callable=check_model_quality, dag=dag)

t_promote = PythonOperator(task_id="promote_model", python_callable=promote_model, dag=dag)
t_reject  = BashOperator(
    task_id="reject_model",
    bash_command='echo "Model rejected: accuracy below threshold" && exit 1',
    dag=dag,
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
t_prepare >> t_train >> t_eval >> t_branch >> [t_promote, t_reject]
