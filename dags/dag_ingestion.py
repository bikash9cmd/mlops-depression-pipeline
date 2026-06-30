"""
DAG: depression_data_ingestion
Purpose: Ingest raw data, validate, and load into MariaDB (Star Schema)
Schedule: Daily at midnight
BCU CMP5366 – Bikash Kushwaha
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# ── Default Args ──────────────────────────────────────────────────────────────
default_args = {
    "owner": "bikash",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# ── DAG ───────────────────────────────────────────────────────────────────────
dag = DAG(
    dag_id="depression_data_ingestion",
    default_args=default_args,
    description="Ingest, validate, and load depression risk data into MariaDB",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "ingestion", "cmp5366"],
)

# ── Task Functions ─────────────────────────────────────────────────────────────
DATA_PATH = "/opt/airflow/data"
RAW_FILE  = f"{DATA_PATH}/raw_depression_data.csv"
CLEAN_FILE = f"{DATA_PATH}/clean_depression_data.csv"


def generate_synthetic_data(**context):
    """
    Generate synthetic depression risk dataset.
    In production this would pull from a real source (API/S3/SFTP).
    """
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "age": np.random.randint(22, 60, n),
        "work_hours_per_week": np.random.normal(45, 10, n).clip(20, 90),
        "years_experience": np.random.randint(1, 35, n),
        "job_satisfaction": np.random.randint(1, 11, n),
        "sleep_hours": np.random.normal(6.5, 1.5, n).clip(3, 10),
        "physical_activity_days": np.random.randint(0, 8, n),
        "social_interactions_per_week": np.random.randint(0, 30, n),
        "has_mental_health_support": np.random.randint(0, 2, n),
        "remote_work": np.random.randint(0, 2, n),
        "ingested_at": datetime.utcnow().isoformat(),
    })

    # Synthetic label: high work hours + low sleep + low satisfaction → higher risk
    risk_score = (
        (df["work_hours_per_week"] > 50).astype(int) * 0.3
        + (df["sleep_hours"] < 6).astype(int) * 0.3
        + (df["job_satisfaction"] < 5).astype(int) * 0.2
        + (df["physical_activity_days"] < 2).astype(int) * 0.1
        + (df["has_mental_health_support"] == 0).astype(int) * 0.1
        + np.random.uniform(0, 0.2, n)
    )
    df["depressed"] = (risk_score > 0.5).astype(int)

    os.makedirs(DATA_PATH, exist_ok=True)
    df.to_csv(RAW_FILE, index=False)
    print(f"[INFO] Generated {n} records → {RAW_FILE}")
    return n


def validate_data(**context):
    """Basic data quality checks before processing."""
    df = pd.read_csv(RAW_FILE)

    checks = {
        "no_nulls": df.isnull().sum().sum() == 0,
        "positive_age": (df["age"] >= 18).all(),
        "valid_hours": df["work_hours_per_week"].between(0, 100).all(),
        "valid_sleep": df["sleep_hours"].between(0, 12).all(),
        "valid_label": df["depressed"].isin([0, 1]).all(),
        "min_rows": len(df) >= 100,
    }

    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise ValueError(f"Data validation failed: {failed}")

    print(f"[INFO] All validation checks passed. Shape: {df.shape}")
    context["ti"].xcom_push(key="row_count", value=len(df))


def clean_and_transform(**context):
    """Clean data and apply feature engineering."""
    df = pd.read_csv(RAW_FILE)

    # Clip outliers
    df["work_hours_per_week"] = df["work_hours_per_week"].clip(20, 80)
    df["sleep_hours"]         = df["sleep_hours"].clip(3, 10)

    # Feature engineering
    df["work_life_balance"] = df["sleep_hours"] / (df["work_hours_per_week"] / 7)
    df["activity_sleep_ratio"] = df["physical_activity_days"] / (df["sleep_hours"] + 1)

    df.to_csv(CLEAN_FILE, index=False)
    print(f"[INFO] Cleaned data saved → {CLEAN_FILE}")


def load_to_mariadb(**context):
    """Load cleaned data into MariaDB fact table."""
    import sqlalchemy
    engine = sqlalchemy.create_engine(
        f"mysql+mysqlclient://mlops:mlops123@mariadb:3306/depression_db"
    )
    df = pd.read_csv(CLEAN_FILE)
    df["loaded_at"] = datetime.utcnow()
    df.to_sql("fact_depression_survey", engine, if_exists="append", index=False)
    print(f"[INFO] Loaded {len(df)} rows into MariaDB")


def update_dimension_tables(**context):
    """Update date and demographic dimension tables (Star Schema)."""
    import sqlalchemy
    engine = sqlalchemy.create_engine(
        "mysql+mysqlclient://mlops:mlops123@mariadb:3306/depression_db"
    )
    today = datetime.utcnow().date()
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("""
            INSERT IGNORE INTO dim_date (date_key, year, month, day, quarter)
            VALUES (:dk, :y, :m, :d, :q)
        """), {
            "dk": today.strftime("%Y%m%d"),
            "y": today.year,
            "m": today.month,
            "d": today.day,
            "q": (today.month - 1) // 3 + 1,
        })
        conn.commit()
    print("[INFO] Dimension tables updated.")


# ── Tasks ─────────────────────────────────────────────────────────────────────
t_generate = PythonOperator(
    task_id="generate_synthetic_data",
    python_callable=generate_synthetic_data,
    dag=dag,
)

t_validate = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    dag=dag,
)

t_clean = PythonOperator(
    task_id="clean_and_transform",
    python_callable=clean_and_transform,
    dag=dag,
)

t_load = PythonOperator(
    task_id="load_to_mariadb",
    python_callable=load_to_mariadb,
    dag=dag,
)

t_dims = PythonOperator(
    task_id="update_dimension_tables",
    python_callable=update_dimension_tables,
    dag=dag,
)

t_done = BashOperator(
    task_id="log_completion",
    bash_command='echo "Ingestion DAG completed at $(date)"',
    dag=dag,
)

# ── Pipeline Order ─────────────────────────────────────────────────────────────
t_generate >> t_validate >> t_clean >> t_load >> t_dims >> t_done
