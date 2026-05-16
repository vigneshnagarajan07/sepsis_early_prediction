"""
models.py  —  Sepsis AI · Model Loading & Inference  (v5.2 Simplified)
========================================================================

SIMPLIFICATION: Binary lab classification only.
  has_labs = ANY lab performed (1+ labs) → XGB activates (if >= 2 labs)
  no_labs  = ZERO labs performed → LGBM only

The 4-tier LabScenario logic is REMOVED. Single binary flag used instead.
"""

import os
import json
import logging
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib

log = logging.getLogger("sepsis-api")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", _SCRIPT_DIR)

LGBM_PATH = os.path.join(MODEL_DIR, "lgbm_stream1.txt")
XGB_PATH = os.path.join(MODEL_DIR, "xgb_booster.json")
PLATT_PATH = os.path.join(MODEL_DIR, "platt_scaler.pkl")
PLATT_FUSED_PATH = os.path.join(MODEL_DIR, "platt_fused.json")
REGISTRY_PATH = os.path.join(MODEL_DIR, "feature_registry.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "inference_config.json")

_lgbm_model = None
_xgb_model = None
_platt = None
_platt_fused = None
_lgbm_feats = []
_xgb_feats = []
_cfg = {}


def _load() -> None:
    global _lgbm_model, _xgb_model, _platt, _platt_fused, _lgbm_feats, _xgb_feats, _cfg

    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        _lgbm_feats = reg["lgbm_vital_features"]
        _xgb_feats = reg["xgb_lab_booster_cols"]
        log.info(f"[models] Registry: LGBM={len(_lgbm_feats)} XGB={len(_xgb_feats)}")
    except FileNotFoundError:
        log.warning("[models] feature_registry.json not found")
    except Exception as e:
        log.error(f"[models] registry load error: {e}")

    _cfg = {
        "ensemble": {
            "no_labs": {"lgbm": 1.00, "xgb": 0.00},
            "has_labs": {"lgbm": 0.40, "xgb": 0.60},
        },
        "alert_thresholds": {"red": 0.55, "amber": 0.40},
    }
    try:
        with open(CONFIG_PATH) as f:
            _cfg.update(json.load(f))
        log.info("[models] inference_config.json loaded")
    except FileNotFoundError:
        log.info("[models] inference_config.json absent — using defaults")

    _strict = os.getenv("SEPSIS_STRICT_STARTUP", "1") == "1"

    # LightGBM
    try:
        _lgbm_model = lgb.Booster(model_file=LGBM_PATH)
        stored = _lgbm_model.feature_name()
        if _lgbm_feats and stored != _lgbm_feats:
            msg = f"LightGBM feature mismatch: {len(stored)} vs {len(_lgbm_feats)}"
            if _strict:
                raise RuntimeError(msg)
            log.warning(f"[models] {msg}")
        elif _lgbm_feats:
            log.info(f"[models] LightGBM validated ({len(stored)} features)")
        log.info(f"[models] LightGBM loaded ({_lgbm_model.num_trees()} trees)")
    except RuntimeError:
        raise
    except Exception as e:
        log.error(f"[models] LightGBM load FAILED: {e}")
        if _strict:
            raise RuntimeError(f"LightGBM unavailable: {e}")

    # XGBoost
    try:
        bst = xgb.Booster()
        bst.load_model(XGB_PATH)
        _xgb_model = bst
        if _xgb_feats:
            if bst.num_features() != len(_xgb_feats):
                msg = f"XGB count mismatch: {bst.num_features()} vs {len(_xgb_feats)}"
                if _strict:
                    raise RuntimeError(msg)
                log.warning(f"[models] {msg}")
            else:
                stored_names_attr = bst.attr('feature_names')
                if stored_names_attr:
                    stored_names = stored_names_attr.split(',')
                    if stored_names != _xgb_feats:
                        msg = f"XGB name/order mismatch"
                        if _strict:
                            raise RuntimeError(msg)
                        log.warning(f"[models] {msg}")
                    else:
                        log.info("[models] XGB names+order validated")
                else:
                    log.info(f"[models] XGB count validated; names not stored")
        log.info(f"[models] XGBoost loaded ({bst.num_boosted_rounds()} trees)")
    except RuntimeError:
        raise
    except Exception as e:
        log.error(f"[models] XGBoost load FAILED: {e}")

    # Platt scalers
    try:
        _platt = joblib.load(PLATT_PATH)
        log.info("[models] Platt (LGBM) loaded")
    except FileNotFoundError:
        log.warning("[models] platt_scaler.pkl missing")
    except Exception as e:
        log.error(f"[models] Platt load FAILED: {e}")

    try:
        with open(PLATT_FUSED_PATH) as f:
            _platt_fused = json.load(f)
        log.info("[models] Fused Platt loaded")
    except FileNotFoundError:
        log.warning(
            "[models] platt_fused.json missing — fused scores UNCALIBRATED. "
            "Generate from validation-set fused scores."
        )
    except Exception as e:
        log.error(f"[models] Fused Platt load FAILED: {e}")


_load()


def _lgbm_vec(feat_dict: dict) -> np.ndarray:
    from features import LGBM_FEATURES as FB
    cols = _lgbm_feats if _lgbm_feats else FB
    return np.array([feat_dict.get(c, 0.0) for c in cols], dtype=np.float64)


def _xgb_vec(feat_dict: dict, tft: float) -> np.ndarray:
    from features import XGB_FEATURES as FB
    cols = _xgb_feats if _xgb_feats else FB
    aug = {**feat_dict, "TFT_Score": tft}
    return np.array([aug.get(c, 0.0) for c in cols], dtype=np.float64)


def _predict_lgbm(vec: np.ndarray) -> float:
    p = _lgbm_model.predict(vec.reshape(1, -1))[0]
    return 0.5 if np.isnan(p) else float(np.clip(p, 0.0, 1.0))


def _predict_xgb(vec: np.ndarray) -> float:
    dmat = xgb.DMatrix(vec.reshape(1, -1))
    p = _xgb_model.predict(dmat)[0]
    if not (0.0 <= float(p) <= 1.0):
        log.warning(f"[models] XGB output {p} outside [0,1] — clipping")
    return 0.5 if np.isnan(p) else float(np.clip(p, 0.0, 1.0))


def _calibrate_tft(tft: float) -> float | None:
    if _platt is None:
        return None
    cal = _platt.predict_proba([[tft]])[0][1]
    return float(np.clip(cal, 0.01, 0.99))


def _calibrate_fused(raw_score: float) -> tuple[float, bool]:
    if _platt_fused is None:
        return raw_score, False
    try:
        coef = np.array(_platt_fused["coef"])
        intercept = np.array(_platt_fused["intercept"])
        z = raw_score * float(coef[0][0]) + float(intercept[0])
        prob = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(prob, 0.01, 0.99)), True
    except Exception as e:
        log.error(f"[models] Fused calibration failed: {e}")
        return raw_score, False


def _monitoring_urgency(v: dict, prob: float) -> list[float]:
    urgency = min(1.0, prob * 1.4)
    w = np.array([
        0.05 + 0.02 * (1 - urgency),
        0.07 + 0.02 * (1 - urgency),
        0.12 + 0.10 * urgency,
        0.13 + 0.08 * urgency,
        0.18 + 0.10 * urgency,
        0.30 + 0.20 * urgency,
    ])
    if float(v.get("resp", 18)) >= 20:
        w[2] += 0.06
    if float(v.get("o2sat", 98)) < 95:
        w[4] += 0.07
    w /= w.sum()
    return [round(float(x), 4) for x in w]


def _tropical_mimic(feat_dict: dict) -> dict:
    dns1 = feat_dict.get("Dengue_NS1", 0)
    mrdt = feat_dict.get("Malaria_RDT", 0)
    plat = feat_dict.get("Platelets", 150)
    wbc = feat_dict.get("WBC", 7.0)
    lact = feat_dict.get("Lactate", 1.5)
    qsf = feat_dict.get("qSOFA", 0)

    if dns1 > 0 and plat < 80 and wbc < 4.5 and lact < 2.5 and qsf < 2:
        return {"is_mimic": True, "mimic_type": "dengue", "suppression": 0.60}
    if dns1 > 0 and plat < 150 and wbc < 5.0 and lact < 2.0:
        return {"is_mimic": True, "mimic_type": "dengue_mild", "suppression": 0.80}
    if mrdt > 0 and plat < 120 and lact < 2.5 and qsf < 2:
        return {"is_mimic": True, "mimic_type": "malaria", "suppression": 0.75}
    return {"is_mimic": False, "mimic_type": "none", "suppression": 1.0}


def _deescalation_cap(feat_dict: dict) -> float | None:
    urine = feat_dict.get("UrineOutput", 0.0)
    temp = feat_dict.get("Temp", 38.0)
    lact = feat_dict.get("Lactate", 2.0)
    qsf = feat_dict.get("qSOFA", 0)
    d_lac = feat_dict.get("Delta_3h_Lactate", 0)
    d_cre = feat_dict.get("Delta_3h_Creatinine", 0)
    d_hr = feat_dict.get("Delta_3h_HR", 0)

    if (qsf == 0 and urine >= 1.0 and temp < 38.0 and lact < 2.0
            and d_lac <= 0 and d_cre <= 0 and d_hr <= 0):
        return 0.30
    return None


def predict(lgbm_input: np.ndarray | None,
            xgb_input: np.ndarray | None,
            feat_dict: dict,
            payload: dict) -> dict:

    if _lgbm_model is None:
        raise RuntimeError("LightGBM model not loaded")

    v = payload["vitals"]

    # Stream 1: LightGBM
    vec1 = _lgbm_vec(feat_dict) if _lgbm_feats else lgbm_input
    tft = _predict_lgbm(vec1)

    # Binary lab classification (simplified)
    n_labs = int(feat_dict.get("N_Labs_Drawn", 0))
    has_labs = n_labs >= 1
    no_labs = not has_labs

    # Stream 2: XGBoost (needs 2+ labs for reliable fusion)
    xgb_score = tft
    xgb_ran = False
    XGB_MIN_LABS = 2

    if (has_labs and _xgb_model is not None
            and bool(_xgb_feats) and n_labs >= XGB_MIN_LABS):
        vec2 = _xgb_vec(feat_dict, tft)
        xgb_score = _predict_xgb(vec2)
        xgb_ran = True

    # Fusion
    cfg_ens = _cfg.get("ensemble", {})
    if xgb_ran:
        w_lgbm = cfg_ens.get("has_labs", {}).get("lgbm", 0.40)
        w_xgb = cfg_ens.get("has_labs", {}).get("xgb", 0.60)
    else:
        w_lgbm, w_xgb = 1.00, 0.00

    raw_score = float(np.clip(w_lgbm * tft + w_xgb * xgb_score, 0.01, 0.99))

    # Calibrate fused score
    ai_score, is_fused_calibrated = _calibrate_fused(raw_score)
    if not is_fused_calibrated:
        ai_score = raw_score

    # Platt on TFT (display only)
    calibrated_tft = _calibrate_tft(tft)
    calibrated_flag = calibrated_tft is not None

    # Clinical values
    qsofa = int(feat_dict.get("qSOFA", 0))
    lact = feat_dict.get("Lactate", 1.5)
    plat = feat_dict.get("Platelets", 230.0)
    wbc = feat_dict.get("WBC", 8.5)
    oligo = feat_dict.get("Oliguria", 0)
    urine = feat_dict.get("UrineOutput", 0.8)
    dns1 = feat_dict.get("Dengue_NS1", 0)
    mrdt = feat_dict.get("Malaria_RDT", 0)
    hr = feat_dict.get("HR", 80.0)
    temp = feat_dict.get("Temp", 37.0)
    age = feat_dict.get("Age", 40.0)
    referred = feat_dict.get("Referred_Outside", 0)

    # Tropical mimic
    mimic = _tropical_mimic(feat_dict)
    if mimic["is_mimic"] and has_labs:
        ai_score = float(np.clip(ai_score * mimic["suppression"], 0.01, 0.99))

    # De-escalation
    deesc_cap = _deescalation_cap(feat_dict)
    if deesc_cap is not None:
        ai_score = min(ai_score, deesc_cap)

    # Clinical boost rules
    clinical_boost = 0.0
    alert_override = None

    if has_labs:
        if lact >= 4.0:
            clinical_boost = max(clinical_boost, 0.65)
        elif lact >= 2.0 and (qsofa >= 1 or oligo):
            clinical_boost = max(clinical_boost, 0.52)
        if lact >= 5.0:
            alert_override = "critical"
        if mrdt > 0 and qsofa >= 2 and plat < 120:
            clinical_boost = max(clinical_boost, 0.55)
        if age > 65 and lact >= 2.0 and qsofa >= 1:
            clinical_boost = max(clinical_boost, 0.52)

    # Rules regardless of labs
    if urine < 0.3 and oligo and ai_score > 0.30:
        clinical_boost = max(clinical_boost, 0.40)
    if no_labs and referred and hr > 110 and temp > 38.5:
        clinical_boost = max(clinical_boost, 0.52)

    # No-labs safety net
    if no_labs:
        if ai_score >= 0.52 and qsofa >= 2:
            alert_override = alert_override or "critical"
        elif qsofa >= 2:
            alert_override = alert_override or "warning"
        elif qsofa >= 1 and ai_score >= 0.30:
            alert_override = alert_override or "warning"

    # Sensor detached
    if feat_dict.get("SensorDetached", 0) == 1:
        return {
            "aiScore": None,
            "alertLevel": "sensor_error",
            "message": "Sensor detached — cannot produce reliable prediction. Reattach sensor.",
            "qsofaScore": qsofa,
        }

    if clinical_boost > 0:
        ai_score = float(np.clip(max(ai_score, clinical_boost), 0.01, 0.99))

    # Alert thresholds
    th = _cfg.get("alert_thresholds", {})
    t_red = th.get("red", 0.55)
    t_amber = th.get("amber", 0.40)
    if no_labs:
        t_red -= 0.05
        t_amber -= 0.05

    _RANK = {"none": 0, "warning": 1, "critical": 2}
    if ai_score >= t_red and qsofa >= 2:
        _th_alert = "critical"
    elif ai_score >= t_amber or qsofa >= 2:
        _th_alert = "warning"
    else:
        _th_alert = "none"

    if alert_override:
        alert = (alert_override
                 if _RANK.get(alert_override, 0) >= _RANK.get(_th_alert, 0)
                 else _th_alert)
    else:
        alert = _th_alert

    # Confidence score
    has_history = feat_dict.get("DeltaSourceReadings", 0) > 1
    _CONF = {
        "has_labs": (1.00, 0.82),
        "no_labs": (0.55, 0.40),
    }
    lab_key = "has_labs" if has_labs else "no_labs"
    confidence_score = _CONF[lab_key][0 if has_history else 1]

    # Data quality warnings
    dq_warnings = []
    if no_labs:
        dq_warnings.append(
            "No labs drawn — score based on vitals only (LightGBM). "
            "Draw stat labs (Lactate, WBC, Creatinine) to enable full model."
        )
    elif n_labs == 1:
        dq_warnings.append(
            f"Only 1 lab drawn — clinical rules applied but XGB fusion "
            f"requires 2+ labs. Consider drawing additional labs."
        )
    if not is_fused_calibrated:
        dq_warnings.append(
            "Fused score is UNCALIBRATED — alert thresholds may not reflect true probability. "
            "Generate platt_fused.json from validation set."
        )
    if feat_dict.get("MissingHistory"):
        dq_warnings.append(
            "Insufficient history — delta features set to 0. "
            "Score is conservative until 2+ readings are available."
        )
    if not feat_dict.get("DeltaValid", 0) and feat_dict.get("DeltaSourceReadings", 0) > 1:
        actual_hrs = feat_dict.get("DeltaActualHours", 0)
        dq_warnings.append(
            f"Delta window is {actual_hrs:.1f}h (not true 3h) — "
            "trend information may be incomplete."
        )
    if mimic["is_mimic"] and has_labs:
        dq_warnings.append(
            f"Tropical mimic detected ({mimic['mimic_type']}). "
            "Sepsis score suppressed. Treat per tropical disease protocol."
        )
    if feat_dict.get("CRT_Synthetic"):
        dq_warnings.append(
            "CRT is estimated from MAP+ShockIndex (not measured at bedside). "
            "Provide real CRT when available."
        )
    if deesc_cap is not None:
        dq_warnings.append(
            "De-escalation applied: improving vitals capped score at low-risk."
        )

    # qsofaComponents from raw vitals
    rr_val = float(v.get("resp", 0))
    gcs_val = float(v.get("gcs", 15))
    sbp_val = float(v.get("systolicBp", 120))

    return {
        "aiScore": round(ai_score, 4),
        "qsofaScore": qsofa,
        "qsofaComponents": {
            "rr_ge22": int(rr_val >= 22),
            "gcs_lt15": int(gcs_val < 15),
            "sbp_le100": int(sbp_val <= 100),
        },
        "alertLevel": alert,
        "monitoringUrgencyWeights": _monitoring_urgency(v, ai_score),
        "modelScores": {
            "lgbm_tft": round(tft, 4),
            "xgb_lab": round(xgb_score, 4) if xgb_ran else None,
            "fused_raw": round(raw_score, 4),
            "calibrated_tft": round(calibrated_tft, 4) if calibrated_tft else None,
            "fused_calibrated": is_fused_calibrated,
        },
        "mimicInfo": mimic,
        "confidenceScore": round(confidence_score, 2),
        "dataQualityWarnings": dq_warnings,
        "featureSummary": {
            "crt": round(feat_dict["CRT"], 2),
            "crtSynthetic": feat_dict.get("CRT_Synthetic", True),
            "shockIndex": round(feat_dict["ShockIndex"], 3),
            "hasLabs": has_labs,
            "nLabsDrawn": n_labs,
            "xgbActive": xgb_ran,
            "calibrated": calibrated_flag,
            "fusedCalibrated": is_fused_calibrated,
        },
    }


def health() -> dict:
    platt_ok = _platt is not None
    reg_ok = bool(_lgbm_feats and _xgb_feats)
    fused_ok = _platt_fused is not None
    return {
        "lgbm_loaded": _lgbm_model is not None,
        "xgb_loaded": _xgb_model is not None,
        "platt_loaded": platt_ok,
        "fused_platt_loaded": fused_ok,
        "registry_loaded": reg_ok,
        "models_ready": _lgbm_model is not None,
        "fully_ready": _lgbm_model is not None and platt_ok and reg_ok and fused_ok,
        "warnings": (
            (["platt_scaler.pkl missing"] if not platt_ok else []) +
            (["platt_fused.json missing"] if not fused_ok else []) +
            (["feature_registry.json missing"] if not reg_ok else [])
        ),
    }
