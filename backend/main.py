"""
main.py  –  Sepsis AI · FastAPI Backend
Run:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

import features as feat_eng
import models   as model_eng

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("sepsis-api")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request schema
# ─────────────────────────────────────────────────────────────────────────────

class LabEntry(BaseModel):
    value:     float = 0.0
    # FIX: default False — omitted lab means not drawn, not drawn-with-value-zero.
    # Lactate=0.0 is physiologically impossible and would falsely lower risk score.
    performed: bool  = False

class Vitals(BaseModel):
    hr:          float = Field(default=85,   ge=30,  le=250)
    map:         float = Field(default=90,   ge=30,  le=160)
    resp:        float = Field(default=18,   ge=4,   le=60)
    temp:        float = Field(default=37.0, ge=34,  le=42)
    o2sat:       float = Field(default=98,   ge=60,  le=100)
    urineOutput: float = Field(default=0.8,  ge=0,   le=5)
    gcs:         float = Field(default=15,   ge=3,   le=15)
    systolicBp:  float = Field(default=120,  ge=50,  le=220)

    # BUG FIX: cross-field validator — MAP must not exceed systolicBp.
    # Physiologically MAP = (SBP + 2×DBP)/3 < SBP always.
    @model_validator(mode="after")
    def check_map_sbp_consistency(self) -> "Vitals":
        if self.map > self.systolicBp:
            raise ValueError(
                f"MAP cannot exceed systolicBp "
                f"(got MAP={self.map}, systolicBp={self.systolicBp}). "
                "Check your inputs."
            )
        return self

class VitalsHistory(BaseModel):
    hr:          list[float] = []
    map:         list[float] = []
    resp:        list[float] = []
    temp:        list[float] = []
    o2sat:       list[float] = []
    urineOutput: list[float] = []
    gcs:         list[float] = []
    systolicBp:  list[float] = []

class Labs(BaseModel):
    lactate:   LabEntry = LabEntry()
    pct:       LabEntry = LabEntry()
    wbc:       LabEntry = LabEntry()
    platelets: LabEntry = LabEntry()
    creatinine:LabEntry = LabEntry()
    bilirubin: LabEntry = LabEntry()
    crp:       LabEntry = LabEntry()
    # Tropical POC tests — not routinely ordered in most Tier 2/3 hospitals.
    # FIX: explicit performed=False so omitting these fields means 'not tested'.
    dengueNS1: LabEntry = LabEntry(value=0, performed=False)
    malariaRDT:LabEntry = LabEntry(value=0, performed=False)

class PreviousLabs(BaseModel):
    lactate:   LabEntry | None = None
    pct:       LabEntry | None = None
    wbc:       LabEntry | None = None
    platelets: LabEntry | None = None
    creatinine:LabEntry | None = None
    bilirubin: LabEntry | None = None

class Demographics(BaseModel):
    age:                 float = Field(default=45, ge=0, le=120)
    gender:              str   = "Male"
    bmi:                 float = Field(default=24.5, ge=10, le=60)
    diabetes:            bool  = False
    ckd:                 bool  = False
    cirrhosis:           bool  = False
    malignancy:          bool  = False
    immunosuppression:   bool  = False
    priorAntibiotics:    bool  = False
    referredFromOutside: bool  = False
    gramNegativeRisk:    bool  = False
    malariaEndemic:      bool  = False
    dengueEndemic:       bool  = False

    # BUG FIX: Pydantic v2 default silently ignores unknown fields.
    # Setting extra='ignore' makes the intent explicit and prevents future
    # breakage if Pydantic tightens its defaults.
    model_config = {"extra": "ignore"}

class PredictRequest(BaseModel):
    vitals:          Vitals
    labs:            Labs
    demographics:    Demographics
    vitalsHistory:   VitalsHistory   = VitalsHistory()
    previousLabs:    PreviousLabs    = PreviousLabs()
    intervalSeconds: float           = Field(default=0.0, ge=0)


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    h = model_eng.health()
    log.info(
        f"Startup  →  LightGBM={h['lgbm_loaded']}  XGBoost={h['xgb_loaded']}  "
        f"Platt={h['platt_loaded']}  Registry={h['registry_loaded']}"
    )
    if not h["models_ready"]:
        log.warning("⚠  LightGBM not loaded — all predictions will fail")
    if not h["platt_loaded"]:
        log.warning("⚠  platt_scaler.pkl missing — raw uncalibrated scores will be served")
    if not h["registry_loaded"]:
        log.warning("⚠  feature_registry.json missing — XGBoost disabled, column order unvalidated")
    if h["fully_ready"]:
        log.info("✓  All models and artifacts loaded — fully ready")
    yield

app = FastAPI(
    title="Sepsis AI — Prediction API",
    version="2.1.0",
    lifespan=lifespan,
)

# BUG FIX: CORS — wildcard is fine for development / LAN clinic installs.
# For production deployments behind an nginx reverse proxy, set
# ALLOWED_ORIGINS="https://your-hospital-domain.in" in the environment.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
_origins = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    # allow_credentials must be False when allow_origins=["*"] (browser spec)
    allow_credentials=bool(_raw_origins),
)


# ─────────────────────────────────────────────────────────────────────────────
# Request logging middleware
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0   = time.perf_counter()
    resp = await call_next(request)
    ms   = (time.perf_counter() - t0) * 1000
    log.info(f"{request.method} {request.url.path}  {resp.status_code}  {ms:.1f}ms")
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health-check used by the frontend and orchestration layer."""
    h    = model_eng.health()
    # models_ready = can serve predictions (LGBM at minimum)
    # fully_ready  = LGBM + Platt + feature registry all loaded
    code = 200 if h["models_ready"] else 503
    status = "ok" if h["fully_ready"] else ("degraded" if h["models_ready"] else "down")
    return JSONResponse(content={"status": status, **h}, status_code=code)


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """
    Main prediction endpoint.
    Returns: aiScore, qsofaScore, alertLevel, attentionWeights,
             modelScores, featureSummary, shapDrivers, deltaInfo.

    BUG FIX: differentiated exception handling —
      • RuntimeError (no models loaded) → 503 Service Unavailable
      • ValueError   (bad input range)  → 422 Unprocessable Entity
      • Anything else                   → 500 with logged traceback
    """
    prev_labs_raw = req.previousLabs.model_dump()
    payload = {
        "vitals":          req.vitals.model_dump(),
        "labs":            req.labs.model_dump(),
        "demographics":    req.demographics.model_dump(),
        "vitalsHistory":   req.vitalsHistory.model_dump(),
        "previousLabs":    {k: v for k, v in prev_labs_raw.items() if v is not None},
        "intervalSeconds": req.intervalSeconds,
    }

    try:
        lgbm_vec, xgb_vec, feat_dict = feat_eng.build_feature_vector(payload)
        result = model_eng.predict(lgbm_vec, xgb_vec, feat_dict, payload)
        result["shapDrivers"] = feat_eng.top_shap_drivers(feat_dict, result["aiScore"])
        result["deltaInfo"] = {
            "sourceReadings":  feat_dict.get("DeltaSourceReadings", 0),
            "intervalSec":     feat_dict.get("DeltaIntervalSec", 0.0),
            "deltaHR":         round(feat_dict["Delta_3h_HR"], 2),
            "deltaMAP":        round(feat_dict["Delta_3h_MAP"], 2),
            "deltaResp":       round(feat_dict["Delta_3h_RespRate"], 2),
            "deltaTemp":       round(feat_dict["Delta_3h_Temp"], 3),
            "deltaLactate":    round(feat_dict["Delta_3h_Lactate"], 3),
            "deltaCreatinine": round(feat_dict["Delta_3h_Creatinine"], 3),
        }
        return result

    except RuntimeError as e:
        # No models loaded — service is degraded
        log.error(f"Model unavailable: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    except ValueError as e:
        # Bad clinical values slipped past Pydantic (e.g. NaN, infinite)
        log.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        # Unexpected error — log full traceback for debugging
        log.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")


@app.get("/api/features")
async def feature_names():
    """Returns the actual training feature lists from feature_registry.json.
    Falls back to hardcoded lists if registry is absent.
    Use for debugging column order and paper appendix verification."""
    from models import _lgbm_feats, _xgb_feats
    return {
        "source":        "registry" if _lgbm_feats else "hardcoded_fallback",
        "lgbm_features": _lgbm_feats if _lgbm_feats else feat_eng.LGBM_FEATURES,
        "xgb_features":  _xgb_feats  if _xgb_feats  else feat_eng.XGB_FEATURES,
        "lgbm_count":    len(_lgbm_feats) if _lgbm_feats else len(feat_eng.LGBM_FEATURES),
        "xgb_count":     len(_xgb_feats)  if _xgb_feats  else len(feat_eng.XGB_FEATURES),
    }
