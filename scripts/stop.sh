#!/usr/bin/env bash
# Gracefully stop the MLOps pipeline
echo "[INFO] Stopping MLOps Depression Pipeline..."
docker-compose down
echo "[OK] All services stopped."
