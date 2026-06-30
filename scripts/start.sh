#!/usr/bin/env bash
# ============================================================
# MLOps Depression Pipeline – Startup Script
# BCU CMP5366 – Bikash Kushwaha
# ============================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "============================================================"
echo "  MLOps Depression Prediction Pipeline"
echo "  BCU CMP5366 – Bikash Kushwaha"
echo "============================================================"
echo ""

# ── Pre-flight checks ──────────────────────────────────────────
command -v docker        &>/dev/null || error "Docker not installed."
command -v docker-compose &>/dev/null || error "Docker Compose not installed."

info "Pre-flight checks passed."

# ── Create required local directories ─────────────────────────
mkdir -p data models logs
success "Local directories ready."

# ── Tear down any existing stack ───────────────────────────────
info "Stopping any existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# ── Pull latest images ─────────────────────────────────────────
info "Pulling Docker images (this may take a few minutes on first run)..."
docker-compose pull

# ── Build custom images ────────────────────────────────────────
info "Building FastAPI image..."
docker-compose build fastapi

# ── Initialise database ────────────────────────────────────────
info "Starting MariaDB & Redis..."
docker-compose up -d mariadb redis

info "Waiting 30s for MariaDB to be ready..."
sleep 30

# ── Initialise Airflow DB ──────────────────────────────────────
info "Initialising Airflow metadata DB..."
docker-compose run --rm airflow-init
success "Airflow DB initialised."

# ── Start all services ─────────────────────────────────────────
info "Starting all services..."
docker-compose up -d

echo ""
info "Waiting 60s for all services to stabilise..."
sleep 60

# ── Health checks ──────────────────────────────────────────────
echo ""
info "Running health checks..."

check_service() {
    local name=$1; local url=$2
    if curl -sf "$url" &>/dev/null; then
        success "$name is UP  →  $url"
    else
        warn "$name may still be starting → $url"
    fi
}

check_service "FastAPI"          "http://localhost:8000/health"
check_service "Airflow"          "http://localhost:8080/health"
check_service "Prometheus"       "http://localhost:9090/-/healthy"
check_service "Grafana"          "http://localhost:3000/api/health"

echo ""
echo "============================================================"
echo -e "  ${GREEN}Pipeline is running!${NC}"
echo ""
echo "  Service URLs:"
echo "  ┌─────────────────────────────────────────────────────"
echo "  │  FastAPI Docs   →  http://localhost:8000/docs"
echo "  │  Airflow UI     →  http://localhost:8080  (admin/admin)"
echo "  │  Prometheus     →  http://localhost:9090"
echo "  │  Grafana        →  http://localhost:3000  (admin/admin)"
echo "  └─────────────────────────────────────────────────────"
echo ""
echo "  Next steps:"
echo "  1. Open Airflow UI and enable the DAGs"
echo "  2. Trigger 'depression_data_ingestion' DAG first"
echo "  3. Trigger 'depression_model_training' DAG"
echo "  4. Test prediction: POST http://localhost:8000/predict"
echo "  5. Check drift:     GET  http://localhost:8000/drift"
echo "============================================================"
