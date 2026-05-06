#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Sepsis AI — Backend Startup
# Run this from the sepsis-backend/ directory.
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Optional: activate a virtualenv if present ───────────────────────────────
if [ -d "venv" ]; then
    echo "[start] Activating venv"
    source venv/bin/activate
fi

# ── Install / upgrade dependencies ───────────────────────────────────────────
echo "[start] Installing requirements…"
pip install -q -r requirements.txt

# ── XGBoost feature-order gate ───────────────────────────────────────────────
# Once you verify the XGBoost feature order from your training script, set:
#   export XGB_FEATURE_ORDER_CONFIRMED=1
# before running this script to enable XGBoost in the ensemble.
export XGB_FEATURE_ORDER_CONFIRMED="${XGB_FEATURE_ORDER_CONFIRMED:-0}"
echo "[start] XGB_FEATURE_ORDER_CONFIRMED=${XGB_FEATURE_ORDER_CONFIRMED}"

# ── Launch uvicorn ────────────────────────────────────────────────────────────
echo "[start] Starting FastAPI on http://0.0.0.0:8000"
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
