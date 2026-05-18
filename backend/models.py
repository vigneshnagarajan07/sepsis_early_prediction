"""
models.py  –  Sepsis AI · Model Loading & Inference

Architecture matches the training notebook exactly:
  Stream 1 : LightGBM (wearable vitals)          → TFT_Score
  Stream 2 : XGBoost  (lab booster, has_lab only) → lab_score
  Fusion   : no_labs → 100% LGBM
             has_labs → 40% LGBM + 60% XGB        (training notebook weights)
  Calibrate: Platt scaler fitted on val TFT_Score → final probability
  Alert    : RED ≥ 0.55, AMBER ≥ 0.40             (training notebook thresholds)

Files required in MODEL_DIR:
  lgbm_stream1.txt      LightGBM wearable model
  xgb_booster.json      XGBoost lab booster
  platt_scaler.pkl      sklearn LogisticRegression (Platt scaler)
  feature_registry.json {"lgbm_vital_features": [...], "xgb_lab_booster_cols": [...]}
  inference_config.json (optional override for weights/thresholds)
"""

import os, json, logging
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib

log = logging.getLogger("sepsis-api")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SECURITY S5: Set MODEL_DIR env var to a path OUTSIDE the web root in production.
# Default (same dir as main.py) is fine for dev but exposes weights if
# path traversal in patient_id is not stopped at the patient_stream layer.
# The path traversal fix in patient_stream.py is the primary defence.
MODEL_DIR   = os.getenv("MODEL_DIR", _SCRIPT_DIR)

LGBM_PATH     = os.path.join(MODEL_DIR, "lgbm_stream1.txt")
XGB_PATH      = os.path.join(MODEL_DIR, "xgb_booster.json")
PLATT_PATH    = os.path.join(MODEL_DIR, "platt_scaler.pkl")
REGISTRY_PATH = os.path.join(MODEL_DIR, "feature_registry.json")
CONFIG_PATH   = os.path.join(MODEL_DIR, "inference_config.json")


# ─────────────────────────────────────────────────────────────────────────────
# Global model singletons
# ─────────────────────────────────────────────────────────────────────────────

_lgbm_model   = None   # lgb.Booster
_xgb_model    = None   # xgb.Booster  (loaded from XGBClassifier save)
_platt        = None   # sklearn LogisticRegression
_lgbm_feats   = []     # VITAL_FEAT list from training
_xgb_feats    = []     # LAB_BOOSTER_COLS list from training
_cfg          = {}     # inference_config.json contents


def _load() -> None:
    global _lgbm_model, _xgb_model, _platt, _lgbm_feats, _xgb_feats, _cfg

    # ── feature_registry.json ────────────────────────────────────────────────
    # This is the ground-truth feature list saved by the training notebook.
    # Without it, column order cannot be guaranteed.
    try:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        _lgbm_feats = reg["lgbm_vital_features"]
        _xgb_feats  = reg["xgb_lab_booster_cols"]
        log.info(f"[models] Feature registry: "
                 f"LGBM={len(_lgbm_feats)} cols, XGB={len(_xgb_feats)} cols")
    except FileNotFoundError:
        log.warning(f"[models] feature_registry.json not found at {REGISTRY_PATH}. "
                    "Feature column order cannot be validated. "
                    "Run the training notebook patch cell and copy the file here.")
    except Exception as e:
        log.error(f"[models] feature_registry.json load error: {e}")

    # ── inference_config.json (optional) ─────────────────────────────────────
    _cfg = {
        "ensemble": {
            "no_labs":  {"lgbm": 1.00, "xgb": 0.00},
            "has_labs": {"lgbm": 0.40, "xgb": 0.60},
        },
        "alert_thresholds": {"red": 0.55, "amber": 0.40},
    }
    try:
        with open(CONFIG_PATH) as f:
            _cfg.update(json.load(f))
        log.info("[models] inference_config.json loaded")
    except FileNotFoundError:
        log.info("[models] inference_config.json not found — using training defaults")

    # ── LightGBM ─────────────────────────────────────────────────────────────
    _strict = os.getenv("SEPSIS_STRICT_STARTUP", "0") == "1"
    try:
        # BUG FIX: LightGBM C++ core calls abort() on fatal format errors,
        # killing the whole Python process. We must sanity-check the file
        # header before passing it to lgb.Booster.
        with open(LGBM_PATH, "r") as f:
            first_line = f.readline().strip()
            if not (first_line.startswith("pandas_categorical") or first_line.startswith("version") or first_line.startswith("tree")):
                raise ValueError(f"Invalid LightGBM model header: {first_line[:50]}")

        _lgbm_model = lgb.Booster(model_file=LGBM_PATH)

        # Validate stored feature names match registry
        stored = _lgbm_model.feature_name()
        if _lgbm_feats and stored != _lgbm_feats:
            log.warning(
                f"[models] LightGBM stored {len(stored)} feature names "
                f"but registry has {len(_lgbm_feats)} — possible mismatch!"
            )
        elif _lgbm_feats:
            log.info(f"[models] LightGBM feature names validated ({len(stored)})")

        log.info(f"[models] LightGBM loaded ← {LGBM_PATH}")
    except Exception as e:
        log.error(f"[models] LightGBM load FAILED: {e}")
        if _strict:
            raise RuntimeError(f"SEPSIS_STRICT_STARTUP=1 and LightGBM failed: {e}")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    # The XGBClassifier was trained with .values (numpy) so xgb_booster.json
    # has no stored feature names. We use raw numpy arrays at inference in
    # the exact same column order as LAB_BOOSTER_COLS.
    try:
        bst = xgb.Booster()
        bst.load_model(XGB_PATH)
        _xgb_model = bst

        # Validate feature count
        if _xgb_feats:
            if bst.num_features() != len(_xgb_feats):
                log.warning(
                    f"[models] XGBoost expects {bst.num_features()} features "
                    f"but registry has {len(_xgb_feats)} — COLUMN ORDER MISMATCH"
                )
            else:
                log.info(f"[models] XGBoost feature count validated ({bst.num_features()})")

        log.info(f"[models] XGBoost loaded ← {XGB_PATH}")
    except Exception as e:
        log.error(f"[models] XGBoost load FAILED: {e}")

    # ── Platt scaler ──────────────────────────────────────────────────────────
    try:
        _platt = joblib.load(PLATT_PATH)
        # Check sklearn version compatibility
        import sklearn as _sk
        _sk_ver = tuple(int(x) for x in _sk.__version__.split(".")[:2])
        if _sk_ver >= (1, 8):
            log.warning(
                f"[models] sklearn {_sk.__version__} — Platt scaler was saved with "
                "sklearn 1.6.1. Minor version difference is usually safe but "
                "re-fit platt_scaler.pkl with current sklearn to eliminate warning."
            )
        log.info(f"[models] Platt scaler loaded ← {PLATT_PATH}")
    except FileNotFoundError:
        log.warning(
            "[models] platt_scaler.pkl not found — calibration disabled. "
            "Run the training notebook patch cell and copy platt_scaler.pkl here."
        )
    except Exception as e:
        log.error(f"[models] Platt scaler load FAILED: {e}")


_load()


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector builder — maps feat_dict to training-order numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

def _lgbm_vec(feat_dict: dict) -> np.ndarray:
    """
    Build the LightGBM input vector using the feature list from feature_registry.json.
    Falls back to the hardcoded list from features.py if registry is absent.
    """
    from features import LGBM_FEATURES as FALLBACK
    cols = _lgbm_feats if _lgbm_feats else FALLBACK
    return np.array([feat_dict.get(c, 0.0) for c in cols], dtype=np.float64)


def _xgb_vec(feat_dict: dict, tft_score: float) -> np.ndarray:
    """
    Build the XGBoost input vector.
    The first column MUST be TFT_Score (the LightGBM prediction) because
    LAB_BOOSTER_COLS starts with ["TFT_Score", ...].
    Uses raw numpy arrays — no column names — to match how XGBClassifier was trained.
    """
    from features import XGB_FEATURES as FALLBACK
    cols = _xgb_feats if _xgb_feats else FALLBACK

    # Inject TFT_Score into feat_dict so the lookup below finds it
    augmented = {**feat_dict, "TFT_Score": tft_score}
    return np.array([augmented.get(c, 0.0) for c in cols], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _predict_lgbm(vec: np.ndarray, return_shap: bool = False) -> float | tuple[float, np.ndarray]:
    """Stream 1: LightGBM wearable prediction → TFT_Score."""
    res = _lgbm_model.predict(vec.reshape(1, -1), pred_contrib=return_shap)
    prob = res[0] if not return_shap else _lgbm_model.predict(vec.reshape(1, -1))[0]
    prob = 0.5 if np.isnan(prob) else float(np.clip(prob, 0.0, 1.0))
    if return_shap:
        return prob, res[0]
    return prob


def _predict_xgb(vec: np.ndarray, return_shap: bool = False) -> float | tuple[float, np.ndarray]:
    """
    Stream 2: XGBoost lab booster.
    """
    dmat = xgb.DMatrix(vec.reshape(1, -1))
    # XGBoost Booster.predict(..., pred_contribs=True) returns a numpy array
    if return_shap:
        shaps = _xgb_model.predict(dmat, pred_contribs=True)
        prob  = _xgb_model.predict(dmat)[0]
    else:
        prob  = _xgb_model.predict(dmat)[0]
        shaps = None

    prob = 0.5 if np.isnan(prob) else float(np.clip(prob, 0.0, 1.0))
    if not (0.0 <= float(prob) <= 1.0):
        log.warning(f"[models] XGB output {prob} is outside [0,1] — clipping.")
    
    if return_shap:
        return prob, shaps[0]
    return prob


def _calibrate(tft_score: float) -> float | None:
    """Apply Platt scaler to TFT_Score. Returns None if scaler unavailable."""
    if _platt is None:
        return None
    cal = _platt.predict_proba([[tft_score]])[0][1]
    return float(np.clip(cal, 0.01, 0.99))


def get_lgbm_features() -> list[str]:
    """Authoritative list of LGBM features from registry or fallback."""
    from features import LGBM_FEATURES as FALLBACK
    return _lgbm_feats if _lgbm_feats else FALLBACK


def get_xgb_features() -> list[str]:
    """Authoritative list of XGBoost features from registry or fallback."""
    from features import XGB_FEATURES as FALLBACK
    return _xgb_feats if _xgb_feats else FALLBACK


def _authentic_attention(shaps: dict) -> list[float]:
    """
    Derives temporal urgency weights from actual model influence.
    Replaces rule-based _synthetic_attention.
    """
    lgbm_shaps = np.array(shaps.get("lgbm", []))
    if len(lgbm_shaps) < 1:
        return [0.166] * 6  # fallback uniform

    # Vitals usually occupy the first N positions in our 6-hour buffer logic.
    # We aggregate the absolute influence of time-varying vitals (HR, MAP, Resp, Temp)
    # and their deltas to show 'where' the model is looking in the sequence.
    # For this prototype, we'll use a simplified mapping:
    # 6 bins representing H-5 to H-0.
    
    # We'll use the ratio of Delta vs Raw vitals influence as a proxy for temporal focus.
    v_idx = [0, 1, 2, 3, 4, 5]  # HR, HRV, SpO2, Temp, MAP, Resp
    d_idx = [11, 12, 13, 14]    # Delta HR, MAP, RespRate, Temp
    
    v_inf = np.abs(lgbm_shaps[v_idx]).sum() if len(lgbm_shaps) > 14 else 0.1
    d_inf = np.abs(lgbm_shaps[d_idx]).sum() if len(lgbm_shaps) > 14 else 0.1
    
    # Scale: more delta influence = more focus on recent changes (H-1, H-0)
    # more raw influence = more focus on steady state (H-5..H-2)
    total = v_inf + d_inf
    v_weight = v_inf / total
    d_weight = d_inf / total
    
    # Create 6 bins
    w = np.array([
        0.10 * v_weight, # H-5
        0.15 * v_weight, # H-4
        0.25 * v_weight, # H-3
        0.50 * v_weight, # H-2
        0.40 * d_weight, # H-1
        0.60 * d_weight  # H-0
    ])
    w /= w.sum()
    return [round(float(x), 4) for x in w]


# ─────────────────────────────────────────────────────────────────────────────
# Public: full prediction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict(lgbm_input: np.ndarray | None,
            xgb_input: np.ndarray | None,
            feat_dict: dict,
            payload: dict) -> dict:
    """
    Full prediction pipeline matching the training notebook exactly.

    Parameters
    ----------
    lgbm_input : pre-built numpy vector (from features.py) — used as fallback
                 if feature_registry.json is absent; otherwise rebuilt here
    xgb_input  : pre-built numpy vector — same fallback behaviour
    feat_dict  : raw feature dict (all named features)
    payload    : original request payload (for vitals context)

    Raises RuntimeError if no models are loaded.
    """
    if _lgbm_model is None:
        raise RuntimeError("LightGBM model not loaded")

    v = payload["vitals"]

    # ── Stream 1: LightGBM ───────────────────────────────────────────────────
    vec1  = _lgbm_vec(feat_dict) if _lgbm_feats else lgbm_input
    tft, shap_lgbm = _predict_lgbm(vec1, return_shap=True)

    # ── Stream 2: XGBoost (only when labs are available) ─────────────────────
    no_labs  = bool(feat_dict.get("LabScenario_no_labs", 1))
    has_labs = not no_labs

    xgb_score = tft   # default: no-labs or no-registry → LGBM score only
    shap_xgb  = None

    # SAFETY GATE: only run XGB when:
    #   (a) feature_registry.json is loaded (column order validated)
    #   (b) at least 2 labs are drawn — with only 1 lab, 13/15 XGB lab
    #       features are median-imputed and XGB adds no real signal
    n_labs_drawn = int(feat_dict.get('WBC_Tested',0) + feat_dict.get('Lactate_Tested',0) +
                       feat_dict.get('Creatinine_Tested',0) + feat_dict.get('PCT_Tested',0) +
                       feat_dict.get('Platelets_Tested',0) + feat_dict.get('Bilirubin_Tested',0))
    xgb_min_labs = 2   # minimum drawn labs for XGB to be trusted
    if has_labs and _xgb_model is not None and bool(_xgb_feats) and n_labs_drawn >= xgb_min_labs:
        vec2      = _xgb_vec(feat_dict, tft)
        xgb_score, shap_xgb = _predict_xgb(vec2, return_shap=True)
    elif has_labs and n_labs_drawn == 1:
        # 1 lab drawn: XGB is unreliable; keep LGBM score and flag in warnings
        has_labs  = False   # treat as no-labs for fusion weights
        no_labs   = True

    # ── Fusion (training notebook weights) ───────────────────────────────────
    cfg_ens = _cfg.get("ensemble", {})
    if has_labs and _xgb_model is not None and bool(_xgb_feats) and n_labs_drawn >= xgb_min_labs:
        w_lgbm = cfg_ens.get("has_labs", {}).get("lgbm", 0.40)
        w_xgb  = cfg_ens.get("has_labs", {}).get("xgb",  0.60)
    else:
        w_lgbm = 1.00
        w_xgb  = 0.00

    raw_score = w_lgbm * tft + w_xgb * xgb_score
    raw_score = float(np.clip(raw_score, 0.01, 0.99))

    # ── Platt calibration ─────────────────────────────────────────────────────
    # Platt was fitted ONLY on TFT_Score (pure LGBM output):
    #   platt.fit(val_df[["TFT_Score"]], val_df["SepsisLabel"])
    # Passing raw_score (40% LGBM + 60% XGB) would be outside its training domain.
    #
    # ai_score = raw fused score — MUST match training evaluation:
    #   auroc = roc_auc_score(y_true, final)  ← final is the raw fused score
    # Alert thresholds 0.55 / 0.40 were set on this same raw fused score.
    # calibrated_tft is stored separately for display / paper calibration curve.
    calibrated_tft  = _calibrate(tft)   # Platt(LGBM output only) — display only
    # NaN guard — if both models returned degenerate values, raw_score could be nan
    ai_score        = 0.5 if np.isnan(raw_score) else raw_score
    calibrated_flag = calibrated_tft is not None

    # ── Clinical values ───────────────────────────────────────────────────────
    qsofa = int(feat_dict.get("qSOFA", 0))
    lact  = feat_dict.get("Lactate",    1.5)
    plat  = feat_dict.get("Platelets",  230.0)
    wbc   = feat_dict.get("WBC",        8.5)
    oligo = feat_dict.get("Oliguria",   0)
    urine = feat_dict.get("UrineOutput",0.8)
    dns1  = feat_dict.get("Dengue_NS1", 0)
    mrdt  = feat_dict.get("Malaria_RDT",0)
    hr    = feat_dict.get("HR",         80.0)
    temp  = feat_dict.get("Temp",       37.0)
    age   = feat_dict.get("Age",        40.0)
    referred = feat_dict.get("Referred_Outside", 0)

    # ── Clinical override rules (only when labs available) ───────────────────
    clinical_boost  = 0.0
    alert_override  = None

    if has_labs:
        if lact >= 4.0:
            clinical_boost = max(clinical_boost, 0.65)
        elif lact >= 2.0 and (qsofa >= 1 or oligo):
            clinical_boost = max(clinical_boost, 0.52)
        if lact >= 5.0:
            alert_override = "critical"
        if dns1 > 0 and plat < 50 and wbc < 4.0:
            clinical_boost = max(clinical_boost, 0.70)
            alert_override = "critical"
        elif dns1 > 0 and plat < 150 and wbc < 4.5:
            clinical_boost = max(clinical_boost, 0.52)
        if mrdt > 0 and qsofa >= 1 and plat < 120:
            clinical_boost = max(clinical_boost, 0.55)
        if age > 65 and lact >= 2.0 and qsofa >= 1:
            clinical_boost = max(clinical_boost, 0.52)

    # Rules regardless of labs
    if urine < 0.3 and oligo and ai_score > 0.30:
        clinical_boost = max(clinical_boost, 0.40)
    if no_labs and referred and hr > 110 and temp > 38.5:
        clinical_boost = max(clinical_boost, 0.52)

    # No-labs safety net — conservative clinical rules when blind
    if no_labs:
        if ai_score >= 0.52 and qsofa >= 2:
            alert_override = alert_override or "critical"
        elif qsofa >= 2:
            # Sepsis-3 mandatory: qSOFA ≥ 2 = organ dysfunction suspected
            alert_override = alert_override or "warning"
        elif qsofa >= 1 and ai_score >= 0.30:
            # LOOPHOLE FIX: borderline vitals + qSOFA=1 + score 0.3–0.4 was NONE.
            # A single qSOFA point with a model score above 0.30 warrants monitoring.
            alert_override = alert_override or "warning"

    if clinical_boost > 0:
        ai_score = float(np.clip(max(ai_score, clinical_boost), 0.01, 0.99))

    # ── Alert thresholds (from training notebook) ─────────────────────────────
    thresholds   = _cfg.get("alert_thresholds", {})
    thresh_red   = thresholds.get("red",   0.55)
    thresh_amber = thresholds.get("amber", 0.40)

    # Tighten thresholds when no labs (conservative bias)
    if no_labs:
        thresh_red   = thresh_red   - 0.05
        thresh_amber = thresh_amber - 0.05

    # FIX M3: compute threshold-based alert AND override, take the more severe.
    # Prevents alert_override from LOWERING an alert that thresholds would raise.
    _ALERT_RANK = {"none": 0, "warning": 1, "critical": 2}
    if ai_score >= thresh_red and qsofa >= 2:
        _threshold_alert = "critical"
    elif ai_score >= thresh_amber or qsofa >= 2:
        _threshold_alert = "warning"
    else:
        _threshold_alert = "none"

    if alert_override:
        # Take whichever is MORE severe — override never suppresses a threshold
        alert = (alert_override
                 if _ALERT_RANK.get(alert_override, 0) >= _ALERT_RANK.get(_threshold_alert, 0)
                 else _threshold_alert)
    else:
        alert = _threshold_alert

    # ── Confidence score ──────────────────────────────────────────────────────
    lab_scen    = _label_lab_scenario(feat_dict)
    has_history = feat_dict.get("DeltaSourceReadings", 0) > 1
    _CONF = {
        "full":          (1.00, 0.82),
        "partial_full":  (0.82, 0.68),
        "partial_cbc":   (0.70, 0.58),
        "no_labs":       (0.55, 0.40),
    }
    confidence_score = _CONF[lab_scen][0 if has_history else 1]

    # ── Data quality warnings ─────────────────────────────────────────────────
    # FIX #17: renamed from `warnings` to avoid shadowing stdlib warnings module
    dq_warnings: list[str] = []
    if no_labs:
        dq_warnings.append(
            "No lab results — score is vitals only (LightGBM stream). "
            "XGBoost lab booster not activated. "
            "Draw stat labs to enable full fusion model."
        )
    if not calibrated_flag:
        dq_warnings.append(
            "Platt calibration unavailable (platt_scaler.pkl missing). "
            "Raw uncalibrated probability displayed."
        )
    if not has_history:
        dq_warnings.append(
            "First reading — all 3-hour delta features are 0 (no prior data). "
            "The score is conservative: rising trends in HR, WBC, Lactate, "
            "Creatinine are invisible until reading 2+. "
            "Re-evaluate after the next reading."
        )

    return {
        "aiScore":            round(ai_score, 4),
        "qsofaScore":         qsofa,
        "alertLevel":         alert,
        "attentionWeights":   _authentic_attention({"lgbm": shap_lgbm}),
        "modelScores": {
            "lgbm_tft":     round(tft, 4),
            "xgb_lab":      round(xgb_score, 4) if has_labs else None,
            "fused_raw":    round(raw_score, 4),
            "calibrated":   round(calibrated_tft, 4) if calibrated_tft is not None else None,
        },
        "confidenceScore":     round(confidence_score, 2),
        "dataQualityWarnings": dq_warnings,
        "featureSummary": {
            "crt":            round(feat_dict["CRT"], 2),
            "crtSynthetic":   feat_dict.get("CRT_Synthetic", True),
            "shockIndex":     round(feat_dict["ShockIndex"], 3),
            "labScenario":    lab_scen,
            "hasLabs":        has_labs,
            "calibrated":     calibrated_flag,
            "xgbEligible":    bool(xgb_score != tft),  # did XGB actually run?
        },
        # Internal SHAP contributions (raw margin)
        "_shaps": {
            "lgbm":       shap_lgbm.tolist() if shap_lgbm is not None else [],
            "xgb":        shap_xgb.tolist()  if shap_xgb  is not None else [],
            "lgbm_feats": get_lgbm_features(),
            "xgb_feats":  get_xgb_features(),
            "w_lgbm":     w_lgbm,
            "w_xgb":      w_xgb,
        }
    }


def _label_lab_scenario(feat_dict: dict) -> str:
    # FIX #16: fallthrough must be 'no_labs' not 'full'.
    # If all four flags are 0 (e.g. imputation failure), claiming full-lab
    # confidence (1.00) is clinically dangerous. Safe default = no_labs (0.40).
    if feat_dict.get("LabScenario_full"):         return "full"
    if feat_dict.get("LabScenario_partial_full"): return "partial_full"
    if feat_dict.get("LabScenario_partial_cbc"):  return "partial_cbc"
    return "no_labs"  # safe default: lowest confidence tier


def health() -> dict:
    """Derives status from live model objects at call time — never from stale flags."""
    platt_ok = _platt is not None
    reg_ok   = bool(_lgbm_feats and _xgb_feats)
    return {
        "lgbm_loaded":      _lgbm_model is not None,
        "xgb_loaded":       _xgb_model  is not None,
        "platt_loaded":     platt_ok,
        "registry_loaded":  reg_ok,
        "models_ready":     _lgbm_model is not None,
        "fully_ready":      _lgbm_model is not None and platt_ok and reg_ok,
        "warnings": (
            (["platt_scaler.pkl missing — calibration disabled"] if not platt_ok else []) +
            (["feature_registry.json missing — column order unvalidated"] if not reg_ok else [])
        ),
    }
