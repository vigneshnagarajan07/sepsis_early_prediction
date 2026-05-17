"""
main.py  –  Sepsis AI · FastAPI Backend
Run:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
import logging
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Header, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Annotated, Optional
from pydantic import BaseModel, Field, model_validator

from fastapi.concurrency import run_in_threadpool

import features as feat_eng
import models   as model_eng
import patient_stream as ps

# ─────────────────────────────────────────────────────────────────────────────
# Simple in-process rate limiter (token bucket, per-IP)
# For production: replace with slowapi or an nginx limit_req_zone
_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_WINDOW = 60.0   # seconds
_RATE_LIMIT  = int(os.getenv("PREDICT_RATE_LIMIT", "120"))  # reqs / window

def _check_rate(request: Request) -> None:
    """Raise HTTP 429 if client exceeds PREDICT_RATE_LIMIT requests/minute.
    Supports X-Forwarded-For for correct IP detection behind proxies.
    """
    # FIX: Get real client IP even if behind Nginx/ALB
    xff = request.headers.get("X-Forwarded-For")
    client_ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else "unknown")

    now = time.monotonic()
    # Prune ALL empty buckets occasionally to prevent memory leak from dead IPs
    bucket = _rate_store.get(client_ip, [])
    pruned = [t for t in bucket if now - t < _RATE_WINDOW]
    
    if len(pruned) >= _RATE_LIMIT:
        _rate_store[client_ip] = pruned 
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {_RATE_LIMIT} requests per minute."
        )
    
    pruned.append(now)
    _rate_store[client_ip] = pruned

    # Garbage collect: if we have too many IPs, clear the ones with empty buckets
    if len(_rate_store) > 1000:
        dead_ips = [ip for ip, b in _rate_store.items() if not b or now - b[-1] > _RATE_WINDOW]
        for ip in dead_ips:
            del _rate_store[ip]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("sepsis-api")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request schema
# ─────────────────────────────────────────────────────────────────────────────

class LabEntry(BaseModel):
    value:     float = 0.0
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

    @model_validator(mode="after")
    def check_map_sbp_consistency(self) -> "Vitals":
        if self.map > self.systolicBp:
            raise ValueError(
                f"MAP cannot exceed systolicBp "
                f"(got MAP={self.map}, systolicBp={self.systolicBp})."
            )
        return self

_MAX_HIST = 20

class VitalsHistory(BaseModel):
    hr:          list[float] = Field(default=[], max_length=_MAX_HIST)
    map:         list[float] = Field(default=[], max_length=_MAX_HIST)
    resp:        list[float] = Field(default=[], max_length=_MAX_HIST)
    temp:        list[float] = Field(default=[], max_length=_MAX_HIST)
    o2sat:       list[float] = Field(default=[], max_length=_MAX_HIST)
    urineOutput: list[float] = Field(default=[], max_length=_MAX_HIST)
    gcs:         list[float] = Field(default=[], max_length=_MAX_HIST)
    systolicBp:  list[float] = Field(default=[], max_length=_MAX_HIST)

class Labs(BaseModel):
    lactate:   LabEntry = LabEntry()
    pct:       LabEntry = LabEntry()
    wbc:       LabEntry = LabEntry()
    platelets: LabEntry = LabEntry()
    creatinine:LabEntry = LabEntry()
    bilirubin: LabEntry = LabEntry()
    crp:       LabEntry = LabEntry()
    dengueNS1: LabEntry = LabEntry(value=0, performed=False)
    malariaRDT:LabEntry = LabEntry(value=0, performed=False)

class PreviousLabs(BaseModel):
    lactate:   LabEntry | None = None
    pct:       LabEntry | None = None
    wbc:       LabEntry | None = None
    platelets: LabEntry | None = None
    creatinine:LabEntry | None = None
    bilirubin: LabEntry | None = None
    crp:       LabEntry | None = None

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
    log.info(f"Startup  →  LGBM={h['lgbm_loaded']} XGB={h['xgb_loaded']} Platt={h['platt_loaded']}")
    yield

app = FastAPI(
    title="Sepsis AI — Prediction API",
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-Session-ID", "X-API-Key"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0   = time.perf_counter()
    resp = await call_next(request)
    ms   = (time.perf_counter() - t0) * 1000
    log.info(f"{request.method} {request.url.path}  {resp.status_code}  {ms:.1f}ms")
    return resp


@app.get("/api/health")
async def health():
    h = model_eng.health()
    code = 200 if h["models_ready"] else 503
    return JSONResponse(content={**h}, status_code=code)


@app.post("/api/predict")
async def predict(req: PredictRequest, request: Request):
    _check_rate(request)
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
        result = await run_in_threadpool(model_eng.predict, lgbm_vec, xgb_vec, feat_dict, payload)
        result["shapDrivers"] = feat_eng.top_shap_drivers(feat_dict, result["aiScore"])
        result["deltaInfo"] = {
            "sourceReadings":  feat_dict.get("DeltaSourceReadings", 0),
            "deltaHR":         round(feat_dict.get("Delta_3h_HR", 0.0), 2),
            "deltaMAP":        round(feat_dict.get("Delta_3h_MAP", 0.0), 2),
        }
        return result
    except Exception as e:
        log.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Session API
# ─────────────────────────────────────────────────────────────────────────────

_SESSION_KEY = os.getenv("SESSION_API_KEY", "")

def _check_session_auth(x_api_key: Annotated[Optional[str], Header()] = None) -> None:
    if _SESSION_KEY and x_api_key != _SESSION_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

class SessionStartRequest(BaseModel):
    patient_id: str

@app.get("/api/patients")
async def list_patients():
    return {"patients": ps.list_patients()}

@app.post("/api/session/start")
async def session_start(req: SessionStartRequest, request: Request,
                        background_tasks: BackgroundTasks,
                        x_session_id: Annotated[Optional[str], Header(alias="X-Session-ID")] = None,
                        _auth: None = Depends(_check_session_auth)):
    _check_rate(request)
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-ID")
    background_tasks.add_task(ps.cleanup_sessions)
    try:
        payload = ps.start_session(req.patient_id, x_session_id)
        return await _run_prediction_on_payload(payload)
    except Exception as e:
        log.exception("session/start error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/tick")
async def session_tick(request: Request,
                       x_session_id: Annotated[Optional[str], Header(alias="X-Session-ID")] = None,
                       _auth: None = Depends(_check_session_auth)):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-ID")
    try:
        payload = ps.tick(x_session_id)
        return await _run_prediction_on_payload(payload)
    except StopIteration:
        return {"done": True}
    except Exception as e:
        log.exception("session/tick error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/status")
async def session_status(x_session_id: Annotated[Optional[str], Header(alias="X-Session-ID")] = None):
    if not x_session_id: return {"active": False}
    return ps.session_status(x_session_id)

@app.post("/api/session/stop")
async def session_stop(x_session_id: Annotated[Optional[str], Header(alias="X-Session-ID")] = None):
    if x_session_id: ps.stop_session(x_session_id)
    return {"stopped": True}

async def _run_prediction_on_payload(payload: dict) -> dict:
    lgbm_vec, xgb_vec, feat_dict = feat_eng.build_feature_vector(payload)
    result = await run_in_threadpool(model_eng.predict, lgbm_vec, xgb_vec, feat_dict, payload)
    result["_meta"] = payload["_meta"]
    result["vitals"] = payload["vitals"]
    result["labs"] = payload["labs"]
    result["demographics"] = payload["demographics"]
    return result
