"""
main.py  –  Coimbatore Sepsis AI · FastAPI Backend
Run:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import features as feat_eng
import models   as model_eng

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("sepsis-api")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request schema (mirrors the TypeScript interfaces in App.tsx)
# ─────────────────────────────────────────────────────────────────────────────

class LabEntry(BaseModel):
    value:     float = 0.0
    performed: bool  = True

class Vitals(BaseModel):
    hr:          float = Field(default=85,  ge=30,  le=250)
    map:         float = Field(default=90,  ge=30,  le=160)
    resp:        float = Field(default=18,  ge=4,   le=60)
    temp:        float = Field(default=37.0,ge=34,  le=42)
    o2sat:       float = Field(default=98,  ge=60,  le=100)
    urineOutput: float = Field(default=0.8, ge=0,   le=5)
    gcs:         float = Field(default=15,  ge=3,   le=15)
    systolicBp:  float = Field(default=120, ge=50,  le=220)

class Labs(BaseModel):
    lactate:   LabEntry = LabEntry()
    pct:       LabEntry = LabEntry()
    wbc:       LabEntry = LabEntry()
    platelets: LabEntry = LabEntry()
    creatinine:LabEntry = LabEntry()
    bilirubin: LabEntry = LabEntry()
    dengueNS1: LabEntry = LabEntry(value=0)
    malariaRDT:LabEntry = LabEntry(value=0)

class Demographics(BaseModel):
    age:                 float = Field(default=45, ge=0, le=120)
    gender:              str   = "Male"
    bmi:                 float = Field(default=24.5, ge=10, le=60)
    diabetes:            bool  = False
    ckd:                 bool  = False
    priorAntibiotics:    bool  = False
    referredFromOutside: bool  = False
    gramNegativeRisk:    bool  = False
    malariaEndemic:      bool  = False
    dengueEndemic:       bool  = False
    covidPrevalence:     bool  = False

class PredictRequest(BaseModel):
    vitals:       Vitals
    labs:         Labs
    demographics: Demographics


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    h = model_eng.health()
    log.info(f"Startup  →  XGBoost={h['xgb_loaded']}  LightGBM={h['lgbm_loaded']}")
    if not h["models_ready"]:
        log.warning("⚠  No models loaded — all predictions will fail")
    yield

app = FastAPI(
    title="Coimbatore Sepsis AI — Prediction API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request logging middleware
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0  = time.perf_counter()
    resp = await call_next(request)
    ms  = (time.perf_counter() - t0) * 1000
    log.info(f"{request.method} {request.url.path}  {resp.status_code}  {ms:.1f}ms")
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health-check used by the frontend and orchestration layer."""
    h = model_eng.health()
    code = 200 if h["models_ready"] else 503
    return JSONResponse(content={"status": "ok" if h["models_ready"] else "degraded", **h}, status_code=code)


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """
    Main prediction endpoint.
    Accepts the full clinical payload from the frontend, returns:
      aiScore         – ensemble sepsis probability [0, 1]
      qsofaScore      – 0-3 qSOFA points
      alertLevel      – 'none' | 'warning' | 'critical'
      attentionWeights– 6-step temporal heatmap weights
      modelScores     – per-model probabilities (transparency)
      featureSummary  – derived features for clinical reference
      shapDrivers     – top influential features (surrogate SHAP)
    """
    # Serialise pydantic → plain dicts for downstream functions
    payload = {
        "vitals":       req.vitals.model_dump(),
        "labs":         req.labs.model_dump(),   # model_dump() already recurses into LabEntry
        "demographics": req.demographics.model_dump(),
    }

    try:
        lgbm_vec, xgb_vec, feat_dict = feat_eng.build_feature_vector(payload)
        result = model_eng.predict(lgbm_vec, xgb_vec, feat_dict, payload)
        result["shapDrivers"] = feat_eng.top_shap_drivers(feat_dict, result["aiScore"])
        return result

    except Exception as e:
        log.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/features")
async def feature_names():
    """Returns the feature name lists — useful for debugging / paper appendix."""
    return {
        "lgbm_features": feat_eng.LGBM_FEATURES,
        "xgb_features":  feat_eng.XGB_FEATURES,
    }
