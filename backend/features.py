"""
features.py  —  Sepsis AI · Feature Engineering  (v8 Integrated)
==================================================================

INTEGRATION NOTE:
  LGBM_FEATURES: 36 features (v7) — preserves 4-tier LabScenario one-hots
  to match trained lgbm_stream1.txt. Oliguria is computed but stored in
  feat_dict for clinical rules only (requires retraining to be a model feature).

  Binary lab YES/NO logic for clinical rules uses has_labs_binary flag.
  The 4-tier one-hots are still in the feature vector for model compatibility.

FIXES:
  FE-001  CRT: real bedside measurement takes priority; synthetic fallback.
  FE-002  1-lab case: clinical boost rules apply for ANY performed lab.
  FE-003  CRP counted in lab counting.
  FE-004  Delta validation with accurate lookback hours metadata.
  FE-005  Endemic flags REMOVED (dead code eliminated).
  FE-006  Gender defaults to Unknown.
  FE-009  Oliguria added to LGBM_FEATURES.
"""

import numpy as np
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Feature name lists  (must match training order in feature_registry.json)
# ─────────────────────────────────────────────────────────────────────────────

LGBM_FEATURES = [
    "HR", "HRV_SDNN", "SpO2", "Temp", "MAP", "RespRate",
    "CRT", "ShockIndex", "MotionArtifact", "SensorDetached", "qSOFA",
    "Delta_3h_HR", "Delta_3h_MAP", "Delta_3h_RespRate", "Delta_3h_Temp",
    "Delta_3h_Lactate", "Delta_3h_Creatinine", "Delta_3h_WBC", "Delta_3h_Platelets",
    "Dengue_NS1", "Malaria_RDT",
    "Age", "BMI", "Diabetes", "CKD", "Cirrhosis", "Malignancy",
    "Immunosuppression", "Prior_Unprescribed_Abx", "Referred_Outside",
    "AMR_Resistance", "Gender_M",
    "LabScenario_no_labs", "LabScenario_partial_cbc",
    "LabScenario_partial_full", "LabScenario_full",
]  # 36 features

XGB_EXTRA_FEATURES = [
    "TFT_Score",
    "Lactate", "PCT", "WBC", "Platelets", "Creatinine", "Bilirubin",
    "WBC_Tested", "Lactate_Tested", "Creatinine_Tested",
    "PCT_Tested", "Platelets_Tested", "Bilirubin_Tested",
    "CRP", "CRP_Tested",
]  # 15 extras → 50 total

XGB_FEATURES = LGBM_FEATURES + XGB_EXTRA_FEATURES

_NAN_FALLBACKS: dict[str, float] = {
    "HR": 85.0, "HRV_SDNN": 50.0, "SpO2": 98.0, "Temp": 37.0,
    "MAP": 85.0, "RespRate": 18.0, "CRT": 1.5, "ShockIndex": 0.7,
    "Age": 45.0, "BMI": 24.5, "Lactate": 1.5, "PCT": 0.08,
    "WBC": 8.5, "Platelets": 230.0, "Creatinine": 0.95, "Bilirubin": 0.6,
    "Oliguria": 0.0, "Has_Labs_Binary": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Derived / approximated features
# ─────────────────────────────────────────────────────────────────────────────

def _crt(hr: float, map_: float, shock_index: float) -> float:
    """Synthetic CRT fallback. Real bedside CRT takes priority."""
    base = 1.5
    map_pen = max(0.0, (75.0 - map_) / 20.0)
    si_pen = max(0.0, (shock_index - 0.8) * 0.8)
    return float(np.clip(base + map_pen + si_pen, 0.8, 3.2))


def _hrv_sdnn(hr: float) -> float:
    return float(np.clip(85.0 - (hr - 60.0) * 0.75, 12.0, 80.0))



def _lab_scenario(labs: dict[str, Any]) -> tuple[int, int, int, int]:
    """
    4-tier lab scenario one-hot for LGBM model feature vector.
    MUST stay for model compatibility — lgbm_stream1.txt was trained with these.
    n == 0       → no_labs       (1,0,0,0)
    1 <= n <= 2  → partial_cbc   (0,1,0,0)
    3 <= n <= 4  → partial_full  (0,0,1,0)
    n >= 5       → full          (0,0,0,1)
    CRP counted (FE-003).
    """
    core_keys = ["lactate","pct","wbc","platelets","creatinine","bilirubin","crp"]
    n = sum(1 for k in core_keys if labs.get(k, {}).get("performed", False))
    if n == 0:   return (1, 0, 0, 0)
    elif n <= 2: return (0, 1, 0, 0)
    elif n <= 4: return (0, 0, 1, 0)
    else:        return (0, 0, 0, 1)

def _lab_count(labs: dict[str, Any]) -> int:
    """Count performed labs. CRP included (FE-003)."""
    core_keys = ["lactate", "pct", "wbc", "platelets", "creatinine", "bilirubin", "crp"]
    return sum(1 for k in core_keys if labs.get(k, {}).get("performed", False))


def _impute_lab(lab: dict[str, Any], key: str) -> float:
    """Return lab value if performed; else training-set median."""
    MEDIANS = {
        "lactate": 1.5, "pct": 0.08, "wbc": 8.5,
        "platelets": 230.0, "creatinine": 0.95,
        "bilirubin": 0.6, "crp": 5.0,
    }
    if lab.get("performed", False):
        return float(lab.get("value", MEDIANS.get(key, 0.0)))
    return MEDIANS.get(key, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Delta computation
# ─────────────────────────────────────────────────────────────────────────────

_MIN_INTERVAL_SEC = 60.0
_MIN_DELTA_READINGS = 2


def _baseline_index(history_len: int, interval_sec: float) -> tuple[int, bool, float]:
    """Returns (index, is_valid, actual_hours_back)."""
    if interval_sec < _MIN_INTERVAL_SEC or history_len < _MIN_DELTA_READINGS:
        return 0, False, 0.0
    THREE_HOURS_SEC = 3 * 3600
    steps_back = int(round(THREE_HOURS_SEC / interval_sec))
    if steps_back >= history_len:
        actual_hours = (history_len - 1) * interval_sec / 3600.0
        return 0, False, actual_hours
    idx = history_len - 1 - steps_back
    return max(0, idx), True, 3.0


def _vital_delta(current: float, history: list[float],
                 interval_sec: float = 0.0) -> tuple[float, bool, float]:
    """Returns (delta, is_valid, actual_hours)."""
    if not history or len(history) < _MIN_DELTA_READINGS:
        return 0.0, False, 0.0
    if 0 < interval_sec < _MIN_INTERVAL_SEC:
        return 0.0, False, 0.0
    idx, is_valid, actual_hours = _baseline_index(len(history), interval_sec)
    if not is_valid:
        return 0.0, False, actual_hours
    return float(current - history[idx]), True, actual_hours


def _lab_delta(current_val: float, performed: bool,
               prev_labs: dict, key: str) -> float:
    if not performed:
        return 0.0
    prev = prev_labs.get(key, {})
    if not prev.get("performed", False):
        return 0.0
    return float(current_val - float(prev.get("value", current_val)))


# ─────────────────────────────────────────────────────────────────────────────
# Public: build feature vectors
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector(payload: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns
    -------
    lgbm_vec  : np.ndarray shape (35,)
    xgb_vec   : np.ndarray shape (50,)
    feat_dict : full feature dict
    """
    import logging as _log
    _logger = _log.getLogger("sepsis-api")

    v = payload["vitals"]
    labs = payload["labs"]
    d = payload["demographics"]
    hist = payload.get("vitalsHistory", {})
    ivl = float(payload.get("intervalSeconds", 0.0))
    prev_labs = payload.get("previousLabs", {})

    if 0 < ivl < _MIN_INTERVAL_SEC:
        _logger.debug(f"intervalSeconds={ivl} below {_MIN_INTERVAL_SEC}s — deltas disabled")

    # ── Vitals ───────────────────────────────────────────────────
    hr = float(v["hr"])
    map_ = float(v["map"])
    resp = float(v["resp"])
    temp = float(v["temp"])
    spo2 = float(v["o2sat"])
    systolic_bp = float(v.get("systolicBp", 120.0))
    gcs = float(v.get("gcs", 15.0))
    urine_out = float(v.get("urineOutput", 0.8))
    oliguria = int(urine_out < 0.5)

    # FE-001: real CRT priority
    if "crt" in v and v["crt"] is not None:
        crt = float(np.clip(v["crt"], 0.8, 5.0))
        crt_is_synthetic = False
    else:
        shock_index = hr / max(systolic_bp, 1.0)
        crt = _crt(hr, map_, shock_index)
        crt_is_synthetic = True

    shock_index = hr / max(systolic_bp, 1.0)
    hrv_sdnn = _hrv_sdnn(hr)
    qsofa = int(resp >= 22) + int(gcs < 15) + int(systolic_bp <= 100)

    # ── Deltas ─────────────────────────────────────────────────
    missing_history = False
    delta_hr, dhr_valid, dhr_hrs = _vital_delta(hr, hist.get("hr", []), ivl)
    delta_map, dmap_valid, dmap_hrs = _vital_delta(map_, hist.get("map", []), ivl)
    delta_resp, dresp_valid, dresp_hrs = _vital_delta(resp, hist.get("resp", []), ivl)
    delta_temp, dtemp_valid, dtemp_hrs = _vital_delta(temp, hist.get("temp", []), ivl)
    if not any([dhr_valid, dmap_valid, dresp_valid, dtemp_valid]):
        missing_history = True

    valid_hours = [h for h, valid in zip([dhr_hrs, dmap_hrs, dresp_hrs, dtemp_hrs],
                                          [dhr_valid, dmap_valid, dresp_valid, dtemp_valid]) if valid]
    actual_lookback_hours = min(valid_hours) if valid_hours else 0.0

    # ── Lab deltas ─────────────────────────────────────────────
    def ld(key):
        lab = labs.get(key, {})
        return _lab_delta(_impute_lab(lab, key), lab.get("performed", False), prev_labs, key)

    delta_lactate = ld("lactate")
    delta_creatinine = ld("creatinine")
    delta_wbc = ld("wbc")
    delta_platelets = ld("platelets")

    # ── Tropical / POC ─────────────────────────────────────────
    _dns1 = labs.get("dengueNS1", {"value": 0, "performed": False})
    _mrdt = labs.get("malariaRDT", {"value": 0, "performed": False})
    dengue_ns1 = float(_dns1["value"]) if _dns1.get("performed") else 0.0
    malaria_rdt = float(_mrdt["value"]) if _mrdt.get("performed") else 0.0

    # ── Demographics ───────────────────────────────────────────
    age = float(d["age"])
    bmi = float(d["bmi"])
    diabetes = int(d.get("diabetes", False))
    ckd = int(d.get("ckd", False))
    cirrhosis = int(d.get("cirrhosis", False))
    malignancy = int(d.get("malignancy", False))
    immunosuppressed = int(d.get("immunosuppression", False))
    prior_abx = int(d.get("priorAntibiotics", False))
    referred = int(d.get("referredFromOutside", False))
    amr = int(d.get("gramNegativeRisk", False))

    _gender_raw = d.get("gender", "Unknown")
    gender_m = int(str(_gender_raw).strip().lower() in ("male", "m", "1"))

    # ── Binary lab YES/NO (clinical routing) ────────────────────
    n_labs = _lab_count(labs)
    has_labs_binary = n_labs >= 1

    # ── 4-tier one-hot (required by LGBM model feature vector) ──
    no_labs_t, partial_cbc_t, partial_full_t, full_t = _lab_scenario(labs)

    # ── Raw lab values (XGB only) ────────────────────────────
    def _lab(key): return _impute_lab(labs.get(key, {"performed": False}), key)
    lactate = _lab("lactate")
    pct = _lab("pct")
    wbc = _lab("wbc")
    platelets = _lab("platelets")
    creatinine = _lab("creatinine")
    bilirubin = _lab("bilirubin")

    def _tested(key): return int(labs.get(key, {}).get("performed", False))
    wbc_tested = _tested("wbc")
    lactate_tested = _tested("lactate")
    creatinine_tested = _tested("creatinine")
    pct_tested = _tested("pct")
    platelets_tested = _tested("platelets")
    bilirubin_tested = _tested("bilirubin")

    _crp_lab = labs.get("crp", {"performed": False, "value": 0.0})
    crp = float(_crp_lab.get("value", 0.0)) if _crp_lab.get("performed") else 0.0
    crp_tested = int(_crp_lab.get("performed", False))

    # ── Assemble feat_dict ────────────────────────────────────
    feat_dict: dict[str, Any] = {
        "HR": hr, "HRV_SDNN": hrv_sdnn, "SpO2": spo2,
        "Temp": temp, "MAP": map_, "RespRate": resp,
        "CRT": crt, "ShockIndex": shock_index,
        "MotionArtifact": int(v.get("motionArtifact", 0)),
        "SensorDetached": int(v.get("sensorDetached", 0)),
        "qSOFA": qsofa,

        "Delta_3h_HR": delta_hr,
        "Delta_3h_MAP": delta_map,
        "Delta_3h_RespRate": delta_resp,
        "Delta_3h_Temp": delta_temp,
        "Delta_3h_Lactate": delta_lactate,
        "Delta_3h_Creatinine": delta_creatinine,
        "Delta_3h_WBC": delta_wbc,
        "Delta_3h_Platelets": delta_platelets,

        "Dengue_NS1": dengue_ns1, "Malaria_RDT": malaria_rdt,

        "Age": age, "BMI": bmi,
        "Diabetes": diabetes, "CKD": ckd, "Cirrhosis": cirrhosis,
        "Malignancy": malignancy, "Immunosuppression": immunosuppressed,
        "Prior_Unprescribed_Abx": prior_abx,
        "Referred_Outside": referred, "AMR_Resistance": amr,
        "Gender_M": gender_m,

        # Binary flag (clinical routing)
        "Has_Labs_Binary": int(has_labs_binary),

        # 4-tier one-hot — required by model (do not remove)
        "LabScenario_no_labs":      no_labs_t,
        "LabScenario_partial_cbc":  partial_cbc_t,
        "LabScenario_partial_full": partial_full_t,
        "LabScenario_full":         full_t,

        "Lactate": lactate, "PCT": pct, "WBC": wbc,
        "Platelets": platelets, "Creatinine": creatinine, "Bilirubin": bilirubin,
        "WBC_Tested": wbc_tested,
        "Lactate_Tested": lactate_tested,
        "Creatinine_Tested": creatinine_tested,
        "PCT_Tested": pct_tested,
        "Platelets_Tested": platelets_tested,
        "Bilirubin_Tested": bilirubin_tested,
        "CRP": crp, "CRP_Tested": crp_tested,

        "TFT_Score": 0.0,

        "Oliguria": oliguria,
        "UrineOutput": urine_out,

        # Non-model fields
        "CRT_Synthetic": crt_is_synthetic,
        "N_Labs_Drawn": n_labs,
        "DeltaSourceReadings": len(hist.get("hr", [])),
        "DeltaIntervalSec": ivl,
        "MissingHistory": int(missing_history),
        "DeltaValid": int(any([dhr_valid, dmap_valid, dresp_valid, dtemp_valid])),
        "DeltaActualHours": round(actual_lookback_hours, 2),
    }

    for feat, val in feat_dict.items():
        if isinstance(val, float) and np.isnan(val):
            feat_dict[feat] = _NAN_FALLBACKS.get(feat, 0.0)

    lgbm_vec = np.array([feat_dict[f] for f in LGBM_FEATURES], dtype=np.float64)
    xgb_vec = np.array([feat_dict[f] for f in XGB_FEATURES], dtype=np.float64)

    return lgbm_vec, xgb_vec, feat_dict


# ─────────────────────────────────────────────────────────────────────────────
# Clinical risk factors
# ─────────────────────────────────────────────────────────────────────────────

def top_risk_factors(feat_dict: dict, prob: float) -> list[dict]:
    """Heuristic clinical risk weights (NOT TreeSHAP)."""
    WEIGHTS = {
        "CRT": lambda v, p, syn: (v - 2.0) * 0.9 * p * (0.4 if syn else 1.0),
        "SpO2": lambda v, p, _: (96 - v) * 0.15 * p,
        "AMR_Resistance": lambda v, p, _: v * 0.6 * p,
        "Temp": lambda v, p, _: abs(v - 37.0) * 0.3 * p,
        "qSOFA": lambda v, p, _: v * 0.25 * p,
        "Delta_3h_WBC": lambda v, p, _: v * 0.18 * p,
        "Delta_3h_Creatinine": lambda v, p, _: v * 0.14 * p,
        "Delta_3h_HR": lambda v, p, _: v * 0.10 * p,
        "Delta_3h_Lactate": lambda v, p, _: v * 0.12 * p,
        "Delta_3h_Platelets": lambda v, p, _: -v * 0.10 * p,
        "Delta_3h_RespRate": lambda v, p, _: v * 0.08 * p,
        "RespRate": lambda v, p, _: max(0, v - 20) * 0.08 * p,
        "Prior_Unprescribed_Abx": lambda v, p, _: v * 0.55 * p,
        "Referred_Outside": lambda v, p, _: v * 0.45 * p,
        "Lactate": lambda v, p, _: max(0, v - 2.0) * 0.25 * p,
        "HR": lambda v, p, _: max(0, v - 100) * 0.04 * p,
        "Diabetes": lambda v, p, _: v * 0.15 * p,
        "Oliguria": lambda v, p, _: v * 0.35 * p,
        "Dengue_NS1": lambda v, p, _: v * 0.20 * p,
        "Malaria_RDT": lambda v, p, _: v * 0.20 * p,
        "CRP": lambda v, p, _: max(0, v - 10) * 0.05 * p,
    }
    scores = []
    is_syn = bool(feat_dict.get("CRT_Synthetic", True))
    for feat, fn in WEIGHTS.items():
        val = feat_dict.get(feat, 0.0)
        influence = fn(float(val), float(prob), is_syn)
        if abs(influence) > 0.001:
            scores.append({
                "feature": feat,
                "value": round(float(val), 3),
                "influence": round(float(influence), 4),
            })
    scores.sort(key=lambda x: abs(x["influence"]), reverse=True)
    return scores[:8]


def top_shap_drivers(feat_dict: dict, prob: float) -> list[dict]:
    return top_risk_factors(feat_dict, prob)
