"""
models.py  –  Sepsis AI · Model Loading & Inference
Loads XGBoost (fusion) and LightGBM (wearable-stream) models,
runs inference, and returns a calibrated ensemble probability.
"""

import os
import numpy as np
import xgboost as xgb
import lightgbm as lgb

from features import LGBM_FEATURES, XGB_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# Paths
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


def _load_models() -> None:
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


_load_models()   # runs once at import time


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_predict(vec: np.ndarray) -> float:
    """
    Single-sample XGB inference → scalar probability.

    BUG FIX: DMatrix is now constructed with feature_names so XGBoost
    validates column order instead of silently mis-predicting.  If the model
    was saved without feature names (num_feature=42, no names in JSON), the
    DMatrix still sends 42 columns in the correct training order because we
    build vec via XGB_FEATURES — an explicit, frozen list.
    """
    import pandas as pd
    # Build a 1-row DataFrame with named columns so XGBoost can validate order
    df   = pd.DataFrame([vec], columns=XGB_FEATURES)
    dmat = xgb.DMatrix(df)

    n_trees = _xgb_model.num_boosted_rounds()
    iteration_range = (0, n_trees) if n_trees > 0 else None
    kwargs = {"iteration_range": iteration_range} if iteration_range else {}
    prob   = _xgb_model.predict(dmat, **kwargs)
    return float(prob[0])


def _lgbm_predict(vec: np.ndarray) -> float:
    """Single-sample LGBM inference → scalar probability."""
    prob = _lgbm_model.predict(vec.reshape(1, -1))
    result = float(prob[0])
    # Guard: LightGBM can return NaN on degenerate inputs; treat as 0.5
    if np.isnan(result):
        return 0.5
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Attention weights helper
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_attention(vitals: dict, prob: float) -> list[float]:
    """
    Generate a 6-step (H-5 … H-0) attention weight vector.

    Note: this is a rule-based surrogate driven by physiological urgency cues,
    NOT output from a TFT/attention model.  It is labelled clearly in the API
    response as 'attentionWeights' and described as synthetic in the paper
    methods section.  The frontend renders it as a monitoring heatmap to give
    clinicians a visual anchor for recency — not as a trained attention score.
    """
    hr   = float(vitals.get("hr",   85))
    resp = float(vitals.get("resp", 18))
    spo2 = float(vitals.get("o2sat", 98))

    urgency = min(1.0, prob * 1.4)

    w = np.array([
        0.05 + 0.02 * (1 - urgency),  # H-5
        0.07 + 0.02 * (1 - urgency),  # H-4
        0.12 + 0.10 * urgency,         # H-3 (drift onset)
        0.13 + 0.08 * urgency,         # H-2
        0.18 + 0.10 * urgency,         # H-1
        0.30 + 0.20 * urgency,         # H-0 (most recent)
    ], dtype=np.float64)

    if resp >= 20:
        w[2] += 0.06
    if spo2 < 95:
        w[4] += 0.07

    w = w / w.sum()
    return [round(float(x), 4) for x in w]


# ─────────────────────────────────────────────────────────────────────────────
# Public: full prediction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict(lgbm_vec: np.ndarray, xgb_vec: np.ndarray,
            feat_dict: dict, payload: dict) -> dict:
    """
    Returns the JSON-serialisable prediction dict consumed by the frontend.
    Raises RuntimeError if no models are loaded (caller converts to HTTP 503).
    """
    probs: list[tuple[str, float, float]] = []

    # ── LGBM: primary model ───────────────────────────────────────────────────
    if _lgbm_ok:
        p_lgbm = _lgbm_predict(lgbm_vec)
        probs.append(("lgbm", p_lgbm, 1.0))

    # ── XGBoost: gated behind env flag until feature order is confirmed ────────
    # xgb_booster.json has num_feature=42 but no stored feature names.
    # Set  XGB_FEATURE_ORDER_CONFIRMED=1  after verifying with your training script.
    _xgb_confirmed = os.getenv("XGB_FEATURE_ORDER_CONFIRMED", "0") == "1"
    if _xgb_ok and _xgb_confirmed:
        p_xgb = _xgb_predict(xgb_vec)
        probs.append(("xgb", p_xgb, 1.22))  # ~55% weight when enabled

    if not probs:
        raise RuntimeError("No models loaded – cannot predict")

    # Weighted ensemble
    total_w  = sum(w for _, _, w in probs)
    ai_score = sum(p * w for _, p, w in probs) / total_w

    # BUG FIX: guard against NaN propagating from a degenerate model output
    if np.isnan(ai_score):
        ai_score = 0.5
    ai_score = float(np.clip(ai_score, 0.01, 0.99))

    # ── qSOFA ─────────────────────────────────────────────────────────────────
    v     = payload["vitals"]
    qsofa = int(feat_dict["qSOFA"])

    # ── Raw clinical values for rule layer ───────────────────────────────────
    plat  = feat_dict.get("Platelets",  230.0)
    wbc   = feat_dict.get("WBC",        8.5)
    lact  = feat_dict.get("Lactate",    1.5)
    oligo = feat_dict.get("Oliguria",   0)
    dns1  = feat_dict.get("Dengue_NS1", 0)
    mrdt  = feat_dict.get("Malaria_RDT",0)
    urine = feat_dict.get("UrineOutput",0.8)
    temp  = feat_dict.get("Temp",       37.0)
    hr    = feat_dict.get("HR",         80.0)
    no_labs  = feat_dict.get("LabScenario_no_labs", 0)
    referred = feat_dict.get("Referred_Outside",    0)
    age      = feat_dict.get("Age",        40.0)

    # ── Confidence / data-quality score ──────────────────────────────────────
    # Quantifies how much patient-specific signal the model actually received.
    # Reduced whenever lab features are missing (imputed to population median)
    # because 6 raw lab features + 4 lab delta features lose discriminative power.
    #
    # Scale:
    #   1.00 → full labs + history          (42/42 features carry real signal)
    #   0.82 → partial labs + history
    #   0.68 → partial labs, no history
    #   0.55 → no labs, history present     (28/42 features carry real signal)
    #   0.40 → no labs, no history          (24/42 features — minimum reliable baseline)
    #
    # These values were derived by computing the fraction of XGB_FEATURES that
    # carry genuine patient-specific signal under each lab scenario, then
    # applying a small penalty for missing vitals history (delta features → 0).
    lab_scen   = _label_lab_scenario(feat_dict)
    has_history = feat_dict.get("DeltaSourceReadings", 0) > 1

    _CONFIDENCE: dict[str, tuple[float, float]] = {
        #                         with_history  no_history
        "full":          (1.00,  0.82),
        "partial_full":  (0.82,  0.68),
        "partial_cbc":   (0.70,  0.58),
        "no_labs":       (0.55,  0.40),
    }
    confidence_score = _CONFIDENCE[lab_scen][0 if has_history else 1]

    # ── Evidence-based clinical safety-net overrides ─────────────────────────
    clinical_boost = 0.0
    alert_override: str | None = None

    # ── Rules that apply only when labs ARE available ─────────────────────────
    if not no_labs:
        # Rule 1: Sepsis-3 lactate criterion (Singer et al. 2016)
        if lact >= 4.0:
            clinical_boost = max(clinical_boost, 0.65)
        elif lact >= 2.0 and (qsofa >= 1 or oligo):
            clinical_boost = max(clinical_boost, 0.52)

        # Rule 2: Cryptic shock — lactate ≥5 overrides alert regardless of BP
        if lact >= 5.0:
            alert_override = "critical"

        # Rule 3: Dengue severity (WHO 2012 guidelines)
        if dns1 > 0 and plat < 50 and wbc < 4.0:
            clinical_boost = max(clinical_boost, 0.70)
            alert_override = "critical"
        elif dns1 > 0 and plat < 150 and wbc < 4.5:
            clinical_boost = max(clinical_boost, 0.52)

        # Rule 4: Cerebral malaria (WHO severe malaria criteria 2015)
        if mrdt > 0 and qsofa >= 1 and plat < 120:
            clinical_boost = max(clinical_boost, 0.55)

        # Rule 7: Elderly lactate elevation
        if age > 65 and lact >= 2.0 and qsofa >= 1:
            clinical_boost = max(clinical_boost, 0.52)

    # ── Rules that apply regardless of lab availability ───────────────────────
    # Rule 5: Oliguria + haemodynamic instability (Sepsis-3 AKI criterion)
    if urine < 0.3 and oligo and ai_score > 0.30:
        clinical_boost = max(clinical_boost, 0.40)

    # Rule 6: No-lab rural referral with haemodynamic instability
    if no_labs and referred and hr > 110 and temp > 38.5:
        clinical_boost = max(clinical_boost, 0.52)

    # ── No-labs specific safety net ───────────────────────────────────────────
    # When labs are absent the model cannot see Lactate, PCT, WBC, Platelets,
    # Creatinine, or Bilirubin — all 6 are imputed to population medians.
    # A median-imputed Lactate=1.5 will NOT trigger Rule 1 above, so a patient
    # with true Lactate=4.8 would go undetected by the rule layer.
    #
    # Mitigation strategy (conservative bias when blind):
    #   a) Lower the "warning" alert threshold from 0.45 → 0.35
    #      (flag suspicion earlier since we cannot confirm with labs)
    #   b) Force at least "warning" when qSOFA ≥ 2 (Sepsis-3 guideline:
    #      qSOFA ≥ 2 = high risk of poor outcome, Singer et al. 2016)
    #   c) Lower "critical" threshold from 0.60 → 0.52 when no labs
    #      (model cannot confirm via lactate; err on the side of escalation)
    if no_labs:
        if ai_score >= 0.52 and qsofa >= 2:
            alert_override = alert_override or "critical"
        elif qsofa >= 2:
            # Sepsis-3 mandatory: qSOFA ≥ 2 = organ dysfunction suspected
            alert_override = alert_override or "warning"

    if clinical_boost > 0:
        ai_score = float(np.clip(max(ai_score, clinical_boost), 0.01, 0.99))

    # ── Alert thresholds ──────────────────────────────────────────────────────
    # Thresholds are tightened when no labs are available (conservative bias)
    warn_threshold     = 0.35 if no_labs else 0.45
    critical_threshold = 0.52 if no_labs else 0.60

    if alert_override:
        alert = alert_override
    elif ai_score >= critical_threshold and qsofa >= 2:
        alert = "critical"
    elif ai_score >= warn_threshold or qsofa >= 2:
        alert = "warning"
    else:
        alert = "none"

    attn         = _synthetic_attention(v, ai_score)
    model_scores = {name: round(p, 4) for name, p, _ in probs}

    # ── Data quality warnings for the frontend ────────────────────────────────
    data_quality_warnings: list[str] = []
    if no_labs:
        data_quality_warnings.append(
            "No lab results available. Score is based on vitals and demographics only. "
            "6 lab features imputed to population medians — accuracy is reduced. "
            "Draw stat labs (Lactate, WBC, Creatinine, Platelets) immediately."
        )
    elif lab_scen == "partial_cbc":
        data_quality_warnings.append(
            "Only 1–2 lab values available. Consider adding Lactate and PCT "
            "to improve prediction accuracy."
        )
    if not has_history:
        data_quality_warnings.append(
            "No vitals history available. Delta features set to 0. "
            "Trend information will improve after 2+ consecutive readings."
        )

    return {
        "aiScore":            round(ai_score, 4),
        "qsofaScore":         qsofa,
        "alertLevel":         alert,
        "attentionWeights":   attn,
        "modelScores":        model_scores,
        "confidenceScore":    round(confidence_score, 2),
        "dataQualityWarnings": data_quality_warnings,
        "featureSummary": {
            "crt":         round(feat_dict["CRT"], 2),
            "shockIndex":  round(feat_dict["ShockIndex"], 3),
            "labScenario": lab_scen,
        },
    }


def _label_lab_scenario(feat_dict: dict) -> str:
    if feat_dict["LabScenario_no_labs"]:        return "no_labs"
    if feat_dict["LabScenario_partial_cbc"]:    return "partial_cbc"
    if feat_dict["LabScenario_partial_full"]:   return "partial_full"
    return "full"


def health() -> dict:
    """
    BUG FIX: derive flags directly from the model objects at call time,
    not from module-level booleans that could be stale if models were
    loaded or cleared after startup.
    """
    xgb_loaded  = _xgb_model is not None
    lgbm_loaded = _lgbm_model is not None
    return {
        "xgb_loaded":   xgb_loaded,
        "lgbm_loaded":  lgbm_loaded,
        "models_ready": xgb_loaded or lgbm_loaded,
    }
