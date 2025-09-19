#!/usr/bin/env bash
set -euo pipefail

# Environment setup
export MLFLOW_TRACKING_URI="file:///tmp/mlruns-disabled"
export MPLCONFIGDIR="/tmp/matplotlib"
export PYTHONPATH="/app/src:${PYTHONPATH:-}"

echo "===== Application Startup at $(date +'%Y-%m-%d %H:%M:%S') ====="

# Cleanup function
cleanup() {
    echo "--- Shutting down services gracefully ---"
    if [ ! -z "${FASTAPI_PID:-}" ]; then
        kill -TERM "$FASTAPI_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

if [ "${1:-}" = "pytest" ]; then
    echo "--- Running tests ---"
    exec pytest
else
    echo "--- Starting internal FastAPI service on port 8000 ---"
    uvicorn src.credit_risk_app.main:app --host 0.0.0.0 --port 8000 &

    # Give the API a moment to start up
    sleep 10

    echo "--- Starting public Streamlit dashboard on port 7860 ---"
    exec streamlit run src/credit_risk_app/dashboard.py \
        --server.port 7860 \
        --server.address 0.0.0.0 \
        --browser.gatherUsageStats false \
        --server.headless true \
        --server.fileWatcherType none
fi
