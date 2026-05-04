"""
features.py  –  Coimbatore Sepsis AI · Feature Engineering
Maps the frontend JSON payload to the exact feature vectors expected
by the LGBM (36 features) and XGBoost (42 features) models.
"""

import numpy as np
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Feature name lists (must match training order exactly)
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

# XGBoost appends 6 raw lab values to the LGBM feature set
XGB_EXTRA_FEATURES = [
    "Lactate", "PCT", "WBC", "Platelets", "Creatinine", "Bilirubin"
]  # → total 42 features

XGB_FEATURES = LGBM_FEATURES + XGB_EXTRA_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Helper: derived / approximated features
# ─────────────────────────────────────────────────────────────────────────────

def _crt(hr: float, map_: float, shock_index: float) -> float:
    """
    Capillary Refill Time (seconds).
    Approximated from MAP + ShockIndex because CRT is not captured by the
    current wearable sensor set.  Validated range from training data: [0.8, 3.2].
    """
    base   = 1.5
    map_pen  = max(0.0, (75.0 - map_) / 20.0)      # 0 at MAP≥75, rises below
    si_pen   = max(0.0, (shock_index - 0.8) * 0.8)  # 0 at SI≤0.8, rises above
    return float(np.clip(base + map_pen + si_pen, 0.8, 3.2))


def _hrv_sdnn(hr: float) -> float:
    """
    HRV SDNN (ms) approximated from heart rate.
    In health ≈ 50 ms; drops linearly in sepsis as HR rises.
    Training range: [12, 80].
    """
    sdnn = 85.0 - (hr - 60.0) * 0.75
    return float(np.clip(sdnn, 12.0, 80.0))


def _lab_scenario(labs: dict[str, Any]) -> tuple[int, int, int, int]:
    """
    Classify lab availability into 4 one-hot scenario flags.
    Core clinical labs: lactate, pct, wbc, platelets, creatinine, bilirubin.
    (dengue / malaria are point-of-care tests – tracked separately)
    """
    core_keys = ["lactate", "pct", "wbc", "platelets", "creatinine", "bilirubin"]
    performed = [labs[k]["performed"] for k in core_keys if k in labs]
    n = sum(performed)

    # CBC-tier = wbc + platelets available
    cbc_available = labs.get("wbc", {}).get("performed", False) and \
                    labs.get("platelets", {}).get("performed", False)

    if n == 0:
        return (1, 0, 0, 0)   # no_labs
    elif cbc_available and n <= 2:
        return (0, 1, 0, 0)   # partial_cbc
    elif 3 <= n <= 4:
        return (0, 0, 1, 0)   # partial_full
    else:
        return (0, 0, 0, 1)   # full


def _impute_lab(lab: dict[str, Any], key: str) -> float:
    """
    Return lab value if performed; otherwise return clinical median imputation.
    """
    MEDIANS = {
        "lactate":    1.5,
        "pct":        0.08,
        "wbc":        8.5,
        "platelets":  230.0,
        "creatinine": 0.95,
        "bilirubin":  0.6,
    }
    if lab.get("performed", True):
        return float(lab.get("value", MEDIANS.get(key, 0.0)))
    return MEDIANS.get(key, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Public: build feature vectors
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector(payload: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns
    -------
    lgbm_vec  : np.ndarray shape (36,)
    xgb_vec   : np.ndarray shape (42,)
    feat_dict : plain dict of all features (for logging / SHAP passthrough)
    """
    v   = payload["vitals"]
    labs = payload["labs"]
    d   = payload["demographics"]

    # ── Vitals ──────────────────────────────────────────────────────────────
    hr          = float(v["hr"])
    map_        = float(v["map"])
    resp        = float(v["resp"])
    temp        = float(v["temp"])
    spo2        = float(v["o2sat"])
    systolic_bp = float(v.get("systolicBp", 120.0))
    gcs         = float(v.get("gcs", 15.0))

    shock_index = hr / max(systolic_bp, 1.0)
    crt         = _crt(hr, map_, shock_index)
    hrv_sdnn    = _hrv_sdnn(hr)

    # qSOFA (0-3): RR ≥22, altered mentation (GCS<15), SBP ≤100
    qsofa = int(resp >= 22) + int(gcs < 15) + int(systolic_bp <= 100)

    # ── Lab deltas (single snapshot → zero) ─────────────────────────────────
    delta_hr          = 0.0
    delta_map         = 0.0
    delta_resp        = 0.0
    delta_temp        = 0.0
    delta_lactate     = 0.0
    delta_creatinine  = 0.0
    delta_wbc         = 0.0
    delta_platelets   = 0.0

    # ── Tropical / point-of-care labs ───────────────────────────────────────
    dengue_ns1  = float(labs.get("dengueNS1",  {}).get("value", 0))
    malaria_rdt = float(labs.get("malariaRDT", {}).get("value", 0))

    # ── Demographics ────────────────────────────────────────────────────────
    age         = float(d["age"])
    bmi         = float(d["bmi"])
    diabetes    = int(d.get("diabetes", False))
    ckd         = int(d.get("ckd", False))
    cirrhosis   = 0   # not exposed in current UI
    malignancy  = 0
    immunosuppressed = 0
    prior_abx   = int(d.get("priorAntibiotics", False))
    referred    = int(d.get("referredFromOutside", False))
    amr         = int(d.get("gramNegativeRisk", False))
    gender_m    = int(d.get("gender", "Male") == "Male")

    # ── Lab scenario flags ───────────────────────────────────────────────────
    no_labs, partial_cbc, partial_full, full_labs = _lab_scenario(labs)

    # ── Raw lab values (XGB only) ────────────────────────────────────────────
    lactate    = _impute_lab(labs.get("lactate",    {"performed": False}), "lactate")
    pct        = _impute_lab(labs.get("pct",        {"performed": False}), "pct")
    wbc        = _impute_lab(labs.get("wbc",        {"performed": False}), "wbc")
    platelets  = _impute_lab(labs.get("platelets",  {"performed": False}), "platelets")
    creatinine = _impute_lab(labs.get("creatinine", {"performed": False}), "creatinine")
    bilirubin  = _impute_lab(labs.get("bilirubin",  {"performed": False}), "bilirubin")

    # ── Assemble dicts ───────────────────────────────────────────────────────
    feat_dict = {
        "HR": hr, "HRV_SDNN": hrv_sdnn, "SpO2": spo2,
        "Temp": temp, "MAP": map_, "RespRate": resp,
        "CRT": crt, "ShockIndex": shock_index,
        "MotionArtifact": 0, "SensorDetached": 0,
        "qSOFA": qsofa,
        "Delta_3h_HR": delta_hr, "Delta_3h_MAP": delta_map,
        "Delta_3h_RespRate": delta_resp, "Delta_3h_Temp": delta_temp,
        "Delta_3h_Lactate": delta_lactate, "Delta_3h_Creatinine": delta_creatinine,
        "Delta_3h_WBC": delta_wbc, "Delta_3h_Platelets": delta_platelets,
        "Dengue_NS1": dengue_ns1, "Malaria_RDT": malaria_rdt,
        "Age": age, "BMI": bmi,
        "Diabetes": diabetes, "CKD": ckd,
        "Cirrhosis": cirrhosis, "Malignancy": malignancy,
        "Immunosuppression": immunosuppressed,
        "Prior_Unprescribed_Abx": prior_abx,
        "Referred_Outside": referred, "AMR_Resistance": amr,
        "Gender_M": gender_m,
        "LabScenario_no_labs": no_labs,
        "LabScenario_partial_cbc": partial_cbc,
        "LabScenario_partial_full": partial_full,
        "LabScenario_full": full_labs,
        # raw lab values (XGB extras)
        "Lactate": lactate, "PCT": pct, "WBC": wbc,
        "Platelets": platelets, "Creatinine": creatinine, "Bilirubin": bilirubin,
    }

    lgbm_vec = np.array([feat_dict[f] for f in LGBM_FEATURES], dtype=np.float64)
    xgb_vec  = np.array([feat_dict[f] for f in XGB_FEATURES],  dtype=np.float64)

    return lgbm_vec, xgb_vec, feat_dict


def top_shap_drivers(feat_dict: dict, prob: float) -> list[dict]:
    """
    Lightweight rule-based SHAP surrogate.
    Returns top contributing features with signed influence for the UI.
    (Real SHAP requires re-running the model with the shap library – kept
    optional to avoid latency on every request.)
    """
    # Weights derived from SHAP summary plot (Figure 1) — global importance
    # scaled to local context
    WEIGHTS = {
        "CRT":                   lambda v, p: (v - 2.0) * 0.9 * p,
        "SpO2":                  lambda v, p: (96 - v) * 0.15 * p,
        "AMR_Resistance":        lambda v, p:  v * 0.6 * p,
        "Temp":                  lambda v, p: abs(v - 37.0) * 0.3 * p,
        "qSOFA":                 lambda v, p:  v * 0.25 * p,
        "Delta_3h_WBC":          lambda v, p:  v * 0.05 * p,
        "Delta_3h_Creatinine":   lambda v, p:  v * 0.04 * p,
        "HRV_SDNN":              lambda v, p: (50 - v) * 0.02 * p,
        "RespRate":              lambda v, p: max(0, v - 20) * 0.08 * p,
        "Delta_3h_Lactate":      lambda v, p:  v * 0.04 * p,
        "Prior_Unprescribed_Abx":lambda v, p:  v * 0.55 * p,
        "Referred_Outside":      lambda v, p:  v * 0.45 * p,
        "Lactate":               lambda v, p: max(0, v - 2.0) * 0.25 * p,
        "HR":                    lambda v, p: max(0, v - 100) * 0.04 * p,
        "Diabetes":              lambda v, p:  v * 0.15 * p,
    }
    scores = []
    for feat, fn in WEIGHTS.items():
        val = feat_dict.get(feat, 0.0)
        influence = fn(val, prob)
        if abs(influence) > 0.001:
            scores.append({
                "feature": feat,
                "value":   round(val, 3),
                "shap":    round(influence, 4),
            })

    scores.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return scores[:8]
