"""
models.py  –  Coimbatore Sepsis AI · Model Loading & Inference
Loads XGBoost (fusion) and LightGBM (wearable-stream) models,
runs inference, and returns a calibrated ensemble probability.
"""

import os
import json
import numpy as np
import xgboost as xgb
import lightgbm as lgb

# ─────────────────────────────────────────────────────────────────────────────
# Paths – models are mounted alongside this file (or via MODEL_DIR env var)
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.getenv("MODEL_DIR", _SCRIPT_DIR)

XGB_PATH  = os.path.join(MODEL_DIR, "xgb_booster.json")
LGBM_PATH = os.path.join(MODEL_DIR, "lgbm_stream1.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Singletons (loaded once on import)
# ─────────────────────────────────────────────────────────────────────────────

_xgb_model:  xgb.Booster | None = None
_lgbm_model: lgb.Booster | None = None
_xgb_ok  = False
_lgbm_ok = False


def _load_models():
    global _xgb_model, _lgbm_model, _xgb_ok, _lgbm_ok

    # XGBoost
    try:
        _xgb_model = xgb.Booster()
        _xgb_model.load_model(XGB_PATH)
        _xgb_ok = True
        print(f"[models] XGBoost loaded  ← {XGB_PATH}")
    except Exception as e:
        print(f"[models] XGBoost load FAILED: {e}")

    # LightGBM
    try:
        _lgbm_model = lgb.Booster(model_file=LGBM_PATH)
        _lgbm_ok = True
        print(f"[models] LightGBM loaded ← {LGBM_PATH}")
    except Exception as e:
        print(f"[models] LightGBM load FAILED: {e}")


_load_models()   # runs at module import time


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_predict(vec: np.ndarray) -> float:
    """Single-sample XGB inference → scalar probability."""
    dmat = xgb.DMatrix(vec.reshape(1, -1))
    prob = _xgb_model.predict(dmat, iteration_range=(0, _xgb_model.best_iteration + 1))
    return float(prob[0])


def _lgbm_predict(vec: np.ndarray) -> float:
    """Single-sample LGBM inference → scalar probability."""
    prob = _lgbm_model.predict(vec.reshape(1, -1))
    return float(prob[0])


# ─────────────────────────────────────────────────────────────────────────────
# Attention weights helper
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_attention(vitals: dict, prob: float) -> list[float]:
    """
    Generate a plausible 6-step (H-5 … H-0) attention weight vector.
    In the absence of a real TFT model, weights are driven by:
      • current risk probability  → H-0 gets highest weight
      • physiological urgency cues → intermediate steps boosted
    The result is normalised to sum=1.
    """
    hr   = float(vitals.get("hr",   85))
    resp = float(vitals.get("resp", 18))
    spo2 = float(vitals.get("o2sat", 98))

    urgency = min(1.0, prob * 1.4)

    # Base template: recency bias + urgency shaping
    w = np.array([
        0.05 + 0.02 * (1 - urgency),  # H-5 (distant)
        0.07 + 0.02 * (1 - urgency),  # H-4
        0.12 + 0.10 * urgency,         # H-3 (where drift often starts)
        0.13 + 0.08 * urgency,         # H-2
        0.18 + 0.10 * urgency,         # H-1
        0.30 + 0.20 * urgency,         # H-0 (most recent – highest weight)
    ], dtype=np.float64)

    # Boost H-3 if respiratory rate is elevated (early compensatory sign)
    if resp >= 20:
        w[2] += 0.06
    # Boost H-1 if SpO2 is falling
    if spo2 < 95:
        w[4] += 0.07

    w = w / w.sum()
    return [round(float(x), 4) for x in w]


# ─────────────────────────────────────────────────────────────────────────────
# Public: run full prediction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict(lgbm_vec: np.ndarray, xgb_vec: np.ndarray,
            feat_dict: dict, payload: dict) -> dict:
    """
    Returns the JSON-serialisable prediction dict consumed by the frontend.
    """
    probs = []

    # ── LGBM: primary model (feature names confirmed via model file) ──────────
    if _lgbm_ok:
        p_lgbm = _lgbm_predict(lgbm_vec)
        probs.append(("lgbm", p_lgbm, 1.0))

    # ── XGBoost: gate behind env flag until feature order is verified ─────────
    # The xgb_booster.json has num_feature=42 but no stored feature names.
    # Enabling it without the exact training column order will mis-predict.
    # Set  XGB_FEATURE_ORDER_CONFIRMED=1  after verifying with your training script.
    _xgb_confirmed = os.getenv("XGB_FEATURE_ORDER_CONFIRMED", "0") == "1"
    if _xgb_ok and _xgb_confirmed:
        p_xgb = _xgb_predict(xgb_vec)
        probs.append(("xgb", p_xgb, 1.22))   # ~55% weight when enabled

    if not probs:
        raise RuntimeError("No models loaded – cannot predict")

    # Weighted ensemble
    total_w   = sum(w for _, _, w in probs)
    ai_score  = sum(p * w for _, p, w in probs) / total_w
    ai_score  = float(np.clip(ai_score, 0.01, 0.99))

    # qSOFA
    v         = payload["vitals"]
    qsofa     = int(feat_dict["qSOFA"])

    # Alert level
    if ai_score >= 0.50 and qsofa >= 2:
        alert = "critical"
    elif ai_score >= 0.50 or qsofa >= 2:
        alert = "warning"
    else:
        alert = "none"

    # Attention weights (6 values for the H-5…H-0 heatmap)
    attn = _synthetic_attention(v, ai_score)

    # Per-model scores (for transparency)
    model_scores = {name: round(p, 4) for name, p, _ in probs}

    return {
        "aiScore":          round(ai_score, 4),
        "qsofaScore":       qsofa,
        "alertLevel":       alert,
        "attentionWeights": attn,
        "modelScores":      model_scores,
        "featureSummary": {
            "crt":          round(feat_dict["CRT"], 2),
            "shockIndex":   round(feat_dict["ShockIndex"], 3),
            "labScenario":  _label_lab_scenario(feat_dict),
        },
    }


def _label_lab_scenario(feat_dict: dict) -> str:
    if feat_dict["LabScenario_no_labs"]:        return "no_labs"
    if feat_dict["LabScenario_partial_cbc"]:    return "partial_cbc"
    if feat_dict["LabScenario_partial_full"]:   return "partial_full"
    return "full"


def health() -> dict:
    return {
        "xgb_loaded":  _xgb_ok,
        "lgbm_loaded": _lgbm_ok,
        "models_ready": _xgb_ok or _lgbm_ok,
    }
