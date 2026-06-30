# MLOps Depression Prediction Pipeline
### BCU CMP5366 – Data Management and Machine Learning Operations
**Bikash Kushwaha | bikash9cmd | Kathmandu, Nepal**

---

## Overview

End-to-end MLOps pipeline for predicting depression risk in working professionals.  
Built with **Apache Airflow**, **FastAPI**, **Redis**, **MariaDB**, **Evidently AI**, **Prometheus**, and **Grafana** — fully orchestrated via **Docker Compose**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network (mlops-network)          │
│                                                             │
│  ┌──────────────────┐     ┌──────────────────────────────┐  │
│  │  Apache Airflow  │────▶│  MariaDB (Star Schema)       │  │
│  │  - Webserver     │     │  - fact_depression_survey    │  │
│  │  - Scheduler     │     │  - dim_date / dim_profession │  │
│  │  - Celery Worker │     │  - predictions               │  │
│  └────────┬─────────┘     │  - drift_reports             │  │
│           │               └──────────────────────────────┘  │
│           │ DAGs                                            │
│  ┌────────▼──────────────────────┐                         │
│  │  DAG 1: data ingestion        │                         │
│  │  DAG 2: model training        │                         │
│  │  DAG 3: drift monitoring      │                         │
│  └────────┬──────────────────────┘                         │
│           │                                                 │
│  ┌────────▼─────────┐     ┌──────────────────┐             │
│  │  FastAPI App     │────▶│  Redis           │             │
│  │  /predict        │     │  (cache + broker)│             │
│  │  /drift          │     └──────────────────┘             │
│  │  /metrics        │                                      │
│  └────────┬─────────┘                                      │
│           │                                                 │
│  ┌────────▼─────────┐     ┌──────────────────┐             │
│  │  Prometheus      │────▶│  Grafana         │             │
│  │  (metrics scrape)│     │  (dashboards)    │             │
│  └──────────────────┘     └──────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

---

## Services & Ports

| Service    | Port | Credentials   |
|------------|------|---------------|
| FastAPI    | 8000 | –             |
| Airflow    | 8080 | admin / admin |
| Prometheus | 9090 | –             |
| Grafana    | 3000 | admin / admin |
| MariaDB    | 3306 | mlops / mlops123 |
| Redis      | 6379 | –             |

---

## Quick Start

### Prerequisites
- Docker Desktop (or Docker Engine + Docker Compose)
- 8 GB RAM minimum recommended

### 1. Clone and start

```bash
git clone https://github.com/bikash9cmd/mlops-depression-pipeline.git
cd mlops-depression-pipeline
chmod +x scripts/start.sh scripts/stop.sh
./scripts/start.sh
```

### 2. Run the DAGs (in order)

Open Airflow at http://localhost:8080

1. Enable and trigger **`depression_data_ingestion`**
2. Enable and trigger **`depression_model_training`**
3. Enable **`depression_drift_monitoring`** (runs every 6 hours)

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 32,
    "work_hours_per_week": 60,
    "years_experience": 8,
    "job_satisfaction": 4,
    "sleep_hours": 5.5,
    "physical_activity_days": 1,
    "social_interactions_per_week": 5,
    "has_mental_health_support": false,
    "remote_work": true
  }'

# Check drift report
curl http://localhost:8000/drift

# View prediction history
curl http://localhost:8000/predictions/history?limit=10
```

### 4. Stop

```bash
./scripts/stop.sh
```

---

## Database Schema (Star Schema)

```
         dim_date          dim_profession      dim_demographics
             │                   │                    │
             └───────────────────┼────────────────────┘
                                 │
                    fact_depression_survey
                                 │
                     predictions + drift_reports
                      (application tables)
```

---

## API Endpoints

| Method | Endpoint               | Description                         |
|--------|------------------------|-------------------------------------|
| GET    | `/health`              | Service health check                |
| POST   | `/predict`             | Predict depression risk             |
| GET    | `/drift`               | Latest Evidently AI drift report    |
| GET    | `/metrics`             | Prometheus metrics                  |
| GET    | `/model/reload`        | Hot-reload model from disk          |
| GET    | `/predictions/history` | Recent predictions from MariaDB     |
| GET    | `/docs`                | Swagger UI                          |

---

## Tech Stack

| Layer         | Technology                                      |
|---------------|-------------------------------------------------|
| Orchestration | Apache Airflow 2.8 (CeleryExecutor)             |
| API           | FastAPI + Uvicorn + Pydantic                    |
| Database      | MariaDB 10.11 (Star Schema)                     |
| Cache/Broker  | Redis 7.2                                       |
| ML            | scikit-learn (RandomForest), joblib             |
| Monitoring    | Evidently AI, Prometheus, Grafana               |
| IaC / DevOps  | Docker Compose, Bash                            |

---

## Project Structure

```
mlops-depression-pipeline/
├── docker-compose.yml          # Full stack orchestration
├── dags/
│   ├── dag_ingestion.py        # Data ingestion & validation
│   ├── dag_training.py         # Model training & promotion
│   └── dag_monitoring.py       # Evidently AI drift monitoring
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py                 # FastAPI application
├── scripts/
│   ├── init_db.sql             # MariaDB Star Schema init
│   ├── start.sh                # One-command startup
│   └── stop.sh                 # Graceful shutdown
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── dashboards/
│       └── dashboard.yml
├── data/                       # Auto-created by DAGs
└── models/                     # Auto-created by training DAG
```

---

## Author

**Bikash Kushwaha** – AWS Certified Solutions Architect Associate  
GitHub: [bikash9cmd](https://github.com/bikash9cmd) | LinkedIn: [bikash9cmd](https://linkedin.com/in/bikash9cmd)
