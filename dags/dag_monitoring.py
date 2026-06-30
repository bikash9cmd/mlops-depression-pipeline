"""
DAG: depression_drift_monitoring
Purpose: Run Evidently AI data drift detection and push results to Redis + MariaDB
Schedule: Every 6 hours
BCU CMP5366 – Bikash Kushwaha
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import redis

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "bikash",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="depression_drift_monitoring",
    default_args=default_args,
    description="Evidently AI drift monitoring for depression prediction pipeline",
    schedule_interval="0 */6 * * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "monitoring", "evidently", "cmp5366"],
)

DATA_PATH = "/opt/airflow/data"
DRIFT_THRESHOLD = 0.15  # Trigger retraining above this


def load_reference_and_current(**context):
    """Load reference (training) and current (recent) data for comparison."""
    # Reference: original training data
    ref_df = pd.read_csv(f"{DATA_PATH}/X_train.csv")

    # Current: simulate recent inference data with slight drift
    np.random.seed(int(datetime.utcnow().timestamp()) % 100)
    n = 200
    current_df = pd.DataFrame({
        "age": np.random.randint(25, 65, n),
        # Introduce drift: work hours distribution shifted
        "work_hours_per_week": np.random.normal(55, 12, n).clip(20, 90),
        "years_experience": np.random.randint(1, 35, n),
        "job_satisfaction": np.random.randint(1, 8, n),   # slightly lower satisfaction
        "sleep_hours": np.random.normal(5.8, 1.2, n).clip(3, 10),  # less sleep
        "physical_activity_days": np.random.randint(0, 6, n),
        "social_interactions_per_week": np.random.randint(0, 25, n),
        "has_mental_health_support": np.random.randint(0, 2, n),
        "remote_work": np.random.randint(0, 2, n),
    })

    # Only keep columns that exist in reference
    common_cols = [c for c in ref_df.columns if c in current_df.columns]
    ref_df = ref_df[common_cols]
    current_df = current_df[common_cols]

    ref_df.to_csv(f"{DATA_PATH}/ref_for_drift.csv", index=False)
    current_df.to_csv(f"{DATA_PATH}/current_for_drift.csv", index=False)
    print(f"[INFO] Reference shape: {ref_df.shape}, Current shape: {current_df.shape}")


def run_evidently_drift(**context):
    """Run Evidently AI DataDriftPreset and extract metrics."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

        ref_df     = pd.read_csv(f"{DATA_PATH}/ref_for_drift.csv")
        current_df = pd.read_csv(f"{DATA_PATH}/current_for_drift.csv")

        report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(),
        ])
        report.run(reference_data=ref_df, current_data=current_df)

        report_dict = report.as_dict()

        # Extract summary
        dataset_drift = report_dict["metrics"][1]["result"]
        drift_score   = dataset_drift.get("drift_share", 0)
        drifted_cols  = [
            col for col, info in dataset_drift.get("drift_by_columns", {}).items()
            if info.get("drift_detected", False)
        ]

        result = {
            "drift_detected": drift_score > DRIFT_THRESHOLD,
            "drift_score":    round(drift_score, 4),
            "features_drifted": drifted_cols,
            "timestamp": datetime.utcnow().isoformat(),
            "method": "Evidently AI DataDriftPreset",
        }

        # Save HTML report
        report.save_html(f"{DATA_PATH}/evidently_report.html")

    except ImportError:
        # Fallback: statistical drift using KS test
        from scipy import stats

        ref_df     = pd.read_csv(f"{DATA_PATH}/ref_for_drift.csv")
        current_df = pd.read_csv(f"{DATA_PATH}/current_for_drift.csv")

        drifted_cols = []
        p_values = {}
        for col in ref_df.columns:
            stat, p = stats.ks_2samp(ref_df[col], current_df[col])
            p_values[col] = round(p, 4)
            if p < 0.05:
                drifted_cols.append(col)

        drift_score = len(drifted_cols) / len(ref_df.columns)
        result = {
            "drift_detected": drift_score > DRIFT_THRESHOLD,
            "drift_score":    round(drift_score, 4),
            "features_drifted": drifted_cols,
            "p_values": p_values,
            "timestamp": datetime.utcnow().isoformat(),
            "method": "KS Test (Evidently fallback)",
        }

    print(f"[INFO] Drift score: {result['drift_score']}, Drifted cols: {result['features_drifted']}")
    context["ti"].xcom_push(key="drift_result", value=result)

    # Push to Redis
    r = redis.Redis(host="redis", port=6379, decode_responses=True)
    r.setex("drift:latest", 3600 * 12, json.dumps(result))  # 12-hour TTL
    print("[INFO] Drift report pushed to Redis.")


def log_drift_to_db(**context):
    """Persist drift report to MariaDB for trend analysis."""
    import sqlalchemy

    result = context["ti"].xcom_pull(task_ids="run_evidently_drift", key="drift_result")
    engine = sqlalchemy.create_engine(
        "mysql+mysqlclient://mlops:mlops123@mariadb:3306/depression_db"
    )
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("""
            INSERT INTO drift_reports (drift_score, drift_detected, features_drifted, method, created_at)
            VALUES (:score, :detected, :features, :method, NOW())
        """), {
            "score":    result["drift_score"],
            "detected": int(result["drift_detected"]),
            "features": json.dumps(result["features_drifted"]),
            "method":   result.get("method", "unknown"),
        })
        conn.commit()
    print("[INFO] Drift report persisted to MariaDB.")


def check_drift_threshold(**context):
    """Branch: trigger retraining if drift exceeds threshold."""
    result = context["ti"].xcom_pull(task_ids="run_evidently_drift", key="drift_result")
    if result["drift_detected"]:
        print(f"[ALERT] Drift detected! Score: {result['drift_score']} > {DRIFT_THRESHOLD}")
        return "trigger_retraining_alert"
    print(f"[OK] No significant drift. Score: {result['drift_score']}")
    return "no_action_required"


# ── Tasks ──────────────────────────────────────────────────────────────────────
t_load   = PythonOperator(task_id="load_reference_and_current", python_callable=load_reference_and_current, dag=dag)
t_drift  = PythonOperator(task_id="run_evidently_drift",        python_callable=run_evidently_drift,        dag=dag)
t_log    = PythonOperator(task_id="log_drift_to_db",            python_callable=log_drift_to_db,            dag=dag)
t_branch = BranchPythonOperator(task_id="check_drift_threshold", python_callable=check_drift_threshold,    dag=dag)

t_alert  = BashOperator(
    task_id="trigger_retraining_alert",
    bash_command='echo "[ALERT] Retraining triggered due to data drift at $(date)" | tee -a /opt/airflow/logs/drift_alerts.log',
    dag=dag,
)
t_ok = BashOperator(
    task_id="no_action_required",
    bash_command='echo "[OK] Drift within acceptable range at $(date)"',
    dag=dag,
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
t_load >> t_drift >> t_log >> t_branch >> [t_alert, t_ok]
