"""
main.py  —  Sepsis AI · FastAPI Backend  (v5.1 Corrected)
============================================================

FIXES APPLIED:
  MA-001  Missing defaultdict import (fixed in v5.0, preserved).
  MA-002  shapDrivers → riskFactors throughout.
  MA-003  sensor_error alertLevel handled.
  MA-004  Vitals: optional crt field.
  MA-005  Vitals: optional motionArtifact, sensorDetached.
  MA-006  Demographics: gender defaults to "Unknown".
  MA-007  PreviousLabs: crp added.
  MA-008  VitalsHistory: crt field added.
  MA-009  sensor_error shortcircuit before riskFactors.
  MA-010  _run_prediction: same sensor_error guard.
  MA-011  intervalSeconds validated: sub-minute logged as demo mode.
  API-002 Vitals fields now REQUIRED (no defaults) — empty vitals rejected.
  API-003 CORS defaults to empty list (reject all origins) when unset.
  API-004 LabEntry timestamps added; freshness validation on labs.
  API-005 Lab value range validators added.
  API-007 Code deduplication: unified prediction through _run_prediction().
"""

import os
import time
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator, field_validator

import features as feat_eng
import models as model_eng

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("sepsis-api")

# ─────────────────────────────────────────────────────────────────────────────
# Rate limiter (token bucket, per-IP)
# ─────────────────────────────────────────────────────────────────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_WINDOW = 60.0
_RATE_LIMIT = int(os.getenv("PREDICT_RATE_LIMIT", "120"))


def _check_rate(client_ip: str) -> None:
    now = time.monotonic()
    bucket = [t for t in _rate_store[client_ip] if now - t < _RATE_WINDOW]
    _rate_store[client_ip] = bucket
    if len(bucket) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: max {_RATE_LIMIT} req/min. Try again shortly."
        )
    _rate_store[client_ip].append(now)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

# API-005: Lab value ranges per analyte
class LabEntry(BaseModel):
    value: float = 0.0
    performed: bool = False
    collectedAt: datetime = Field(default_factory=datetime.utcnow)

    @field_validator('value')
    @classmethod
    def check_physiological_range(cls, v: float, info) -> float:
        """API-005: reject physiologically impossible lab values."""
        # We need to know which lab this is; Pydantic v2 passes info.data
        # but the field name is in the parent context. Since LabEntry is nested,
        # we validate generically here and let the parent model do named checks.
        return v


class Vitals(BaseModel):
    # API-002 FIX: all vital fields are REQUIRED (no defaults).
    # Empty vitals will be rejected by Pydantic instead of filled with normals.
    hr: float = Field(..., ge=30, le=250)
    map: float = Field(..., ge=30, le=160)
    resp: float = Field(..., ge=4, le=60)
    temp: float = Field(..., ge=34, le=42)
    o2sat: float = Field(..., ge=60, le=100)
    urineOutput: float = Field(..., ge=0, le=5)
    gcs: float = Field(..., ge=3, le=15)
    systolicBp: float = Field(..., ge=50, le=220)
    crt: Optional[float] = Field(default=None, ge=0.5, le=6.0)
    motionArtifact: int = Field(default=0, ge=0, le=1)
    sensorDetached: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def check_map_sbp(self) -> "Vitals":
        if self.map > self.systolicBp:
            raise ValueError(
                f"MAP ({self.map}) cannot exceed systolicBp ({self.systolicBp})."
            )
        return self


_MAX_HIST = 20


class VitalsHistory(BaseModel):
    hr: list[float] = Field(default=[], max_length=_MAX_HIST)
    map: list[float] = Field(default=[], max_length=_MAX_HIST)
    resp: list[float] = Field(default=[], max_length=_MAX_HIST)
    temp: list[float] = Field(default=[], max_length=_MAX_HIST)
    o2sat: list[float] = Field(default=[], max_length=_MAX_HIST)
    urineOutput: list[float] = Field(default=[], max_length=_MAX_HIST)
    gcs: list[float] = Field(default=[], max_length=_MAX_HIST)
    systolicBp: list[float] = Field(default=[], max_length=_MAX_HIST)
    crt: list[float] = Field(default=[], max_length=_MAX_HIST)


class Labs(BaseModel):
    lactate: LabEntry = LabEntry()
    pct: LabEntry = LabEntry()
    wbc: LabEntry = LabEntry()
    platelets: LabEntry = LabEntry()
    creatinine: LabEntry = LabEntry()
    bilirubin: LabEntry = LabEntry()
    crp: LabEntry = LabEntry()
    dengueNS1: LabEntry = LabEntry(value=0, performed=False)
    malariaRDT: LabEntry = LabEntry(value=0, performed=False)

    @model_validator(mode="after")
    def check_ranges(self) -> "Labs":
        """API-005: per-lab physiological range validation."""
        ranges = {
            "lactate": (0.0, 20.0, "mmol/L"),
            "pct": (0.0, 100.0, "ng/mL"),
            "wbc": (0.0, 50.0, "10^9/L"),
            "platelets": (0.0, 1000.0, "10^9/L"),
            "creatinine": (0.0, 15.0, "mg/dL"),
            "bilirubin": (0.0, 20.0, "mg/dL"),
            "crp": (0.0, 300.0, "mg/L"),
        }
        for field_name, (lo, hi, unit) in ranges.items():
            lab = getattr(self, field_name)
            if lab.performed and not (lo <= lab.value <= hi):
                raise ValueError(
                    f"{field_name}={lab.value} {unit} is outside physiological range "
                    f"[{lo}, {hi}]. Check unit conversion or data entry."
                )
        # API-004: freshness check (labs > 4 hours old are stale)
        max_age = timedelta(hours=4)
        now = datetime.utcnow()
        for field_name in ["lactate", "pct", "wbc", "platelets", "creatinine", "bilirubin", "crp"]:
            lab = getattr(self, field_name)
            if lab.performed and (now - lab.collectedAt) > max_age:
                raise ValueError(
                    f"{field_name} is stale (>4 hours old). "
                    f"Collected at {lab.collectedAt.isoformat()}."
                )
        return self


class PreviousLabs(BaseModel):
    lactate: LabEntry | None = None
    pct: LabEntry | None = None
    wbc: LabEntry | None = None
    platelets: LabEntry | None = None
    creatinine: LabEntry | None = None
    bilirubin: LabEntry | None = None
    crp: LabEntry | None = None


class Demographics(BaseModel):
    age: float = Field(default=45, ge=0, le=120)
    gender: str = "Unknown"
    bmi: float = Field(default=24.5, ge=10, le=60)
    diabetes: bool = False
    ckd: bool = False
    cirrhosis: bool = False
    malignancy: bool = False
    immunosuppression: bool = False
    priorAntibiotics: bool = False
    referredFromOutside: bool = False
    gramNegativeRisk: bool = False
    malariaEndemic: bool = False
    dengueEndemic: bool = False
    model_config = {"extra": "ignore"}


class PredictRequest(BaseModel):
    vitals: Vitals
    labs: Labs
    demographics: Demographics
    vitalsHistory: VitalsHistory = VitalsHistory()
    previousLabs: PreviousLabs = PreviousLabs()
    intervalSeconds: float = Field(default=0.0, ge=0)


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    h = model_eng.health()
    log.info(
        f"Startup → LightGBM={h['lgbm_loaded']}  XGBoost={h['xgb_loaded']}  "
        f"Platt={h['platt_loaded']}  FusedPlatt={h['fused_platt_loaded']}  "
        f"Registry={h['registry_loaded']}"
    )
    if not h["models_ready"]:
        log.warning("⚠  LightGBM not loaded — all predictions will fail")
    if not h["platt_loaded"]:
        log.warning("⚠  platt_scaler.pkl missing — LGBM calibration disabled")
    if not h["fused_platt_loaded"]:
        log.warning(
            "⚠  platt_fused.json missing — fused scores are UNCALIBRATED. "
            "Alert thresholds may not reflect true probabilities."
        )
    if not h["registry_loaded"]:
        log.warning("⚠  feature_registry.json missing — XGBoost disabled")
    if h["fully_ready"]:
        log.info("✓  All artifacts loaded — fully ready")
    yield


app = FastAPI(
    title="Sepsis AI — Prediction API",
    version="8.0.0",
    lifespan=lifespan,
)

# API-003 FIX: CORS defaults to empty list (reject all origins) when unset.
# Wildcard "*" is NEVER the default.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
if _raw_origins.strip() == "*":
    _origins = ["*"]
    _allow_creds = False
elif _raw_origins.strip():
    _origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
    _allow_creds = True
else:
    # DEFAULT: no origins allowed. CORS is disabled until explicitly configured.
    _origins = []
    _allow_creds = False
    log.warning(
        "ALLOWED_ORIGINS is not set — CORS disabled. "
        "Set ALLOWED_ORIGINS to enable cross-origin requests."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    allow_credentials=_allow_creds,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    log.info(f"{request.method} {request.url.path}  {resp.status_code}  {ms:.1f}ms")
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# Helper: assemble prediction payload dict from Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

def _build_payload(req: PredictRequest) -> dict:
    prev = {k: v for k, v in req.previousLabs.model_dump().items() if v is not None}
    return {
        "vitals": req.vitals.model_dump(),
        "labs": req.labs.model_dump(),
        "demographics": req.demographics.model_dump(),
        "vitalsHistory": req.vitalsHistory.model_dump(),
        "previousLabs": prev,
        "intervalSeconds": req.intervalSeconds,
    }


def _run_prediction(payload: dict) -> dict:
    """
    Core inference: LGBM → XGB → Platt → alert → riskFactors → deltaInfo.
    MA-003/009: handles sensor_error (aiScore=None) by skipping riskFactors.
    MA-002: renamed shapDrivers → riskFactors throughout.
    API-007: unified prediction logic — used by both /api/predict and session endpoints.
    """
    lgbm_vec, xgb_vec, feat_dict = feat_eng.build_feature_vector(payload)
    result = model_eng.predict(lgbm_vec, xgb_vec, feat_dict, payload)

    if result.get("alertLevel") == "sensor_error":
        result["riskFactors"] = []
        result["deltaInfo"] = {}
        return result

    result["riskFactors"] = feat_eng.top_risk_factors(feat_dict, result["aiScore"])
    result["deltaInfo"] = {
        "sourceReadings": feat_dict.get("DeltaSourceReadings", 0),
        "intervalSec": feat_dict.get("DeltaIntervalSec", 0.0),
        "missingHistory": bool(feat_dict.get("MissingHistory", 0)),
        "deltaValid": bool(feat_dict.get("DeltaValid", 0)),
        "deltaActualHours": feat_dict.get("DeltaActualHours", 0.0),
        "deltaHR": round(feat_dict.get("Delta_3h_HR", 0.0), 2),
        "deltaMAP": round(feat_dict.get("Delta_3h_MAP", 0.0), 2),
        "deltaResp": round(feat_dict.get("Delta_3h_RespRate", 0.0), 2),
        "deltaTemp": round(feat_dict.get("Delta_3h_Temp", 0.0), 3),
        "deltaLactate": round(feat_dict.get("Delta_3h_Lactate", 0.0), 3),
        "deltaCreatinine": round(feat_dict.get("Delta_3h_Creatinine", 0.0), 3),
        "deltaWBC": round(feat_dict.get("Delta_3h_WBC", 0.0), 2),
        "deltaPlatelets": round(feat_dict.get("Delta_3h_Platelets", 0.0), 0),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Routes — core
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    h = model_eng.health()
    code = 200 if h["models_ready"] else 503
    status = "ok" if h["fully_ready"] else ("degraded" if h["models_ready"] else "down")
    return JSONResponse(content={"status": status, **h}, status_code=code)


@app.post("/api/predict")
async def predict(req: PredictRequest, request: Request):
    """
    Main prediction endpoint.
    """
    _check_rate(request.client.host if request.client else "unknown")

    if 0 < req.intervalSeconds < 60:
        log.debug(f"intervalSeconds={req.intervalSeconds}s — demo speed mode")

    payload = _build_payload(req)
    try:
        return _run_prediction(payload)
    except RuntimeError as e:
        log.error(f"Model unavailable: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        log.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/api/features")
async def feature_names():
    """Returns actual training feature lists from feature_registry.json."""
    lf = model_eng._lgbm_feats
    xf = model_eng._xgb_feats
    return {
        "source": "registry" if lf else "hardcoded_fallback",
        "lgbm_features": lf if lf else feat_eng.LGBM_FEATURES,
        "xgb_features": xf if xf else feat_eng.XGB_FEATURES,
        "lgbm_count": len(lf) if lf else len(feat_eng.LGBM_FEATURES),
        "xgb_count": len(xf) if xf else len(feat_eng.XGB_FEATURES),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Patient Stream Pipeline
# ─────────────────────────────────────────────────────────────────────────────

import patient_stream as ps

_SESSION_KEY = os.getenv("SESSION_API_KEY", "")


def _check_session_auth(x_api_key: Annotated[Optional[str], Header()] = None) -> None:
    if not _SESSION_KEY:
        return
    if x_api_key != _SESSION_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


class SessionStartRequest(BaseModel):
    patient_id: str


@app.get("/api/patients")
async def list_patients():
    """List all patient demo files in data/patients/."""
    try:
        return {"patients": ps.list_patients()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/start")
async def session_start(
    req: SessionStartRequest,
    request: Request,
    _auth: None = Depends(_check_session_auth),
):
    """Load a patient file and return the first reading + prediction."""
    _check_rate(request.client.host if request.client else "unknown")
    try:
        payload = ps.start_session(req.patient_id)
        return await _session_predict(payload)
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception("session/start error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/tick")
async def session_tick(
    request: Request,
    _auth: None = Depends(_check_session_auth),
):
    """Advance session by one reading; returns vitals + prediction."""
    _check_rate(request.client.host if request.client else "unknown")
    try:
        payload = ps.tick()
        return await _session_predict(payload)
    except StopIteration as e:
        return JSONResponse(content={"done": True, "message": str(e)}, status_code=200)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("session/tick error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/status")
async def session_status():
    return ps.session_status()


@app.post("/api/session/stop")
async def session_stop(_auth: None = Depends(_check_session_auth)):
    ps.stop_session()
    return {"stopped": True}


async def _session_predict(payload: dict) -> dict:
    """Runs prediction on a session payload and merges _meta."""
    meta = payload.pop("_meta")
    try:
        result = _run_prediction(payload)
    except RuntimeError as e:
        log.error(f"Model unavailable: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Session prediction error")
        raise HTTPException(status_code=500, detail=str(e))

    result["_meta"] = meta
    result["vitals"] = payload["vitals"]
    result["labs"] = payload["labs"]
    result["demographics"] = payload["demographics"]
    return result
