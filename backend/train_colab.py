"""
COIMBATORE SEPSIS AI — COMPLETE PRODUCTION TRAINING PIPELINE
=============================================================
Single cell. Paste entirely into one Colab cell and run.

Architecture:
  Stream 1 : LightGBM on 36 wearable + demographic features  → TFT_Score
  Stream 2 : XGBoost  on 25 lab + TFT_Score features         → lab_score
  Fusion   : no_labs → 100% LGBM | has_labs → 40% LGBM + 60% XGB
  Calibrate: Platt scaler on val TFT_Score

Saves 5 artifacts (copy all to backend/):
  lgbm_stream1.txt
  xgb_booster.json
  platt_scaler.pkl
  feature_registry.json
  inference_config.json

Feature lists match inference/features.py exactly so
feature_registry.json will validate cleanly on startup.
"""

# ── Install ───────────────────────────────────────────────────────
import subprocess
subprocess.run(
    ["pip", "install", "lightgbm>=4.3.0,<4.4.0",
     "xgboost>=2.0.3", "shap", "imbalanced-learn", "-q"],
    check=False
)

# ── Imports ───────────────────────────────────────────────────────
import json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from collections import Counter
from sklearn.linear_model  import LogisticRegression
from sklearn.calibration   import calibration_curve
from sklearn.metrics       import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix,
)

# ── Paths ─────────────────────────────────────────────────────────
TS_PATH   = "/content/coimbatore_v2_timeseries.csv"
META_PATH = "/content/coimbatore_v2_meta.csv"

# ─────────────────────────────────────────────────────────────────
# FEATURE REGISTRY
# These lists are the single source of truth.
# inference/features.py LGBM_FEATURES and XGB_FEATURES must match.
# ─────────────────────────────────────────────────────────────────

# Stream 1 — LightGBM wearable vitals + demographics (no raw lab values)
VITAL_FEAT = [
    # Wearable vitals
    "HR", "HRV_SDNN", "SpO2", "Temp", "MAP", "RespRate",
    "CRT", "ShockIndex", "MotionArtifact", "SensorDetached", "qSOFA",
    # 3-hour delta features (computed from rolling history)
    "Delta_3h_HR", "Delta_3h_MAP", "Delta_3h_RespRate", "Delta_3h_Temp",
    "Delta_3h_Lactate", "Delta_3h_Creatinine", "Delta_3h_WBC", "Delta_3h_Platelets",
    # Tropical POC tests
    "Dengue_NS1", "Malaria_RDT",
    # Demographics & comorbidities
    "Age", "BMI", "Diabetes", "CKD", "Cirrhosis", "Malignancy",
    "Immunosuppression", "Prior_Unprescribed_Abx", "Referred_Outside",
    "AMR_Resistance", "Gender_M",
    # Lab availability one-hot (which scenario the patient is in)
    "LabScenario_no_labs", "LabScenario_partial_cbc",
    "LabScenario_partial_full", "LabScenario_full",
]  # 36 features

# Stream 2 — XGBoost lab booster (VITAL_FEAT + TFT_Score + raw lab values)
LAB_BOOSTER_COLS = VITAL_FEAT + [
    # LGBM prediction — must be FIRST of the extra cols so inference can inject it
    "TFT_Score",
    # Raw lab values (only available when performed=True)
    "Lactate", "PCT", "WBC", "Platelets", "Creatinine", "Bilirubin",
    # Lab availability flags
    "WBC_Tested", "Lactate_Tested", "Creatinine_Tested",
    "PCT_Tested", "Platelets_Tested", "Bilirubin_Tested",
    # CRP (additional inflammatory marker)
    "CRP", "CRP_Tested",
]  # 36 + 14 = 50 features

print("="*60)
print("  COIMBATORE SEPSIS AI — PRODUCTION TRAINING")
print("="*60)
print(f"  LGBM features : {len(VITAL_FEAT)}")
print(f"  XGB  features : {len(LAB_BOOSTER_COLS)}")

# ═════════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ═════════════════════════════════════════════════════════════════
print("\n[1/7] Loading data...")
df = pd.read_csv(TS_PATH)
print(f"  Timeseries : {df.shape}  — {df['PatientID'].nunique()} patients")

meta = pd.read_csv(META_PATH)
print(f"  Metadata   : {meta.shape}")

# ═════════════════════════════════════════════════════════════════
# STEP 2 — MERGE + ENGINEER
# ═════════════════════════════════════════════════════════════════
print("\n[2/7] Feature engineering...")

df = df.merge(meta, on="PatientID", how="left")
df = df.sort_values(["PatientID", "Timestep"]).reset_index(drop=True)

# ── Lab _Tested flags ─────────────────────────────────────────────
for lab in ["WBC", "Lactate", "Creatinine", "PCT", "Platelets", "Bilirubin", "CRP"]:
    if lab in df.columns:
        df[f"{lab}_Tested"] = (~df[lab].isna()).astype(int)
    else:
        df[lab]           = 0.0
        df[f"{lab}_Tested"] = 0

# ── LabScenario one-hot ───────────────────────────────────────────
def classify_lab_scenario(row):
    core = ["WBC_Tested","Platelets_Tested","Creatinine_Tested",
            "Bilirubin_Tested","Lactate_Tested","PCT_Tested"]
    n = sum(row[c] for c in core if c in row)
    if n == 0:    return "no_labs"
    elif n <= 2:  return "partial_cbc"
    elif n <= 4:  return "partial_full"
    else:         return "full"

df["LabScenario"]            = df.apply(classify_lab_scenario, axis=1)
df["LabScenario_no_labs"]    = (df["LabScenario"] == "no_labs").astype(int)
df["LabScenario_partial_cbc"]= (df["LabScenario"] == "partial_cbc").astype(int)
df["LabScenario_partial_full"]=(df["LabScenario"] == "partial_full").astype(int)
df["LabScenario_full"]       = (df["LabScenario"] == "full").astype(int)

# ── Gender one-hot ────────────────────────────────────────────────
if "Gender" in df.columns:
    df["Gender_M"] = (df["Gender"].astype(str).str.upper() == "M").astype(int)
else:
    df["Gender_M"] = 0

# ── qSOFA (Seymour 2016: SBP ≤ 100, RR ≥ 22, GCS < 15) ──────────
if "SystolicBP" in df.columns:
    df["qSOFA"] = (
        (df.get("RespRate", 0) >= 22).astype(int) +
        (df.get("GCS", 15)    < 15).astype(int)  +
        (df["SystolicBP"]     <= 100).astype(int)
    )
elif "qSOFA" not in df.columns:
    df["qSOFA"] = 0

# ── Derived vitals ────────────────────────────────────────────────
if "ShockIndex" not in df.columns and "SystolicBP" in df.columns:
    df["ShockIndex"] = df["HR"] / df["SystolicBP"].clip(lower=1)

if "CRT" not in df.columns:
    # Approximate CRT from MAP + ShockIndex
    map_   = df.get("MAP", pd.Series(85, index=df.index))
    si_    = df.get("ShockIndex", pd.Series(0.7, index=df.index))
    df["CRT"] = (1.5 +
                 (75 - map_).clip(lower=0) / 20 +
                 (si_ - 0.8).clip(lower=0) * 0.8).clip(0.8, 3.2)

if "HRV_SDNN" not in df.columns:
    df["HRV_SDNN"] = (85 - (df["HR"] - 60) * 0.75).clip(12, 80)

if "MotionArtifact" not in df.columns:
    df["MotionArtifact"] = 0
if "SensorDetached" not in df.columns:
    df["SensorDetached"] = 0

# ── 3-hour delta features ─────────────────────────────────────────
# For each patient, compute the value 3h ago (or earliest available)
# and subtract from current value.
print("  Computing 3-hour delta features...")
DELTA_COLS = {
    "Delta_3h_HR":          "HR",
    "Delta_3h_MAP":         "MAP",
    "Delta_3h_RespRate":    "RespRate",
    "Delta_3h_Temp":        "Temp",
    "Delta_3h_Lactate":     "Lactate",
    "Delta_3h_Creatinine":  "Creatinine",
    "Delta_3h_WBC":         "WBC",
    "Delta_3h_Platelets":   "Platelets",
}
# Assume 5-min Timestep cadence → 3 h = 36 steps back
LOOKBACK = 36

for delta_col, source_col in DELTA_COLS.items():
    if source_col in df.columns:
        df[delta_col] = (
            df.groupby("PatientID")[source_col]
            .transform(lambda s: s - s.shift(LOOKBACK).fillna(s.iloc[0]))
        )
    else:
        df[delta_col] = 0.0

# ── Ensure all required columns exist ────────────────────────────
for col in VITAL_FEAT + LAB_BOOSTER_COLS:
    if col not in df.columns and col != "TFT_Score":
        df[col] = 0.0

df = df.fillna(0)

# ═════════════════════════════════════════════════════════════════
# STEP 3 — AGGREGATE 5-MIN → 15-MIN WINDOWS
# ═════════════════════════════════════════════════════════════════
print("\n[3/7] Aggregating 5-min → 15-min windows...")

df["window_idx"] = df["Timestep"] // 3

agg = {}
MEAN_COLS = [c for c in VITAL_FEAT if c not in
             ["MotionArtifact","SensorDetached","qSOFA",
              "LabScenario_no_labs","LabScenario_partial_cbc",
              "LabScenario_partial_full","LabScenario_full",
              "Dengue_NS1","Malaria_RDT",
              "Diabetes","CKD","Cirrhosis","Malignancy",
              "Immunosuppression","Prior_Unprescribed_Abx",
              "Referred_Outside","AMR_Resistance","Gender_M"]]
MAX_COLS  = ["MotionArtifact","SensorDetached","qSOFA",
             "LabScenario_no_labs","LabScenario_partial_cbc",
             "LabScenario_partial_full","LabScenario_full",
             "Dengue_NS1","Malaria_RDT","SepsisLabel"]
LAST_COLS = ["Lactate","PCT","WBC","Platelets","Creatinine","Bilirubin","CRP",
             "WBC_Tested","Lactate_Tested","Creatinine_Tested",
             "PCT_Tested","Platelets_Tested","Bilirubin_Tested","CRP_Tested"]
FIRST_COLS= ["Diabetes","CKD","Cirrhosis","Malignancy","Immunosuppression",
             "Prior_Unprescribed_Abx","Referred_Outside","AMR_Resistance",
             "Gender_M","Age","BMI","Outcome"]

for c in MEAN_COLS:
    if c in df.columns: agg[c] = "mean"
for c in MAX_COLS:
    if c in df.columns: agg[c] = "max"
for c in LAST_COLS:
    if c in df.columns: agg[c] = "last"
for c in FIRST_COLS:
    if c in df.columns: agg[c] = "first"

df_15 = (df.groupby(["PatientID","window_idx"])
           .agg(agg)
           .reset_index()
           .sort_values(["PatientID","window_idx"])
           .reset_index(drop=True))

df_15["time_idx"] = df_15.groupby("PatientID").cumcount()
df_15 = df_15.fillna(0)

print(f"  15-min shape    : {df_15.shape}")
print(f"  Unique patients : {df_15['PatientID'].nunique()}")
print(f"  SepsisLabel=1   : {df_15['SepsisLabel'].mean():.1%}")

# ═════════════════════════════════════════════════════════════════
# STEP 4 — PATIENT-WISE STRATIFIED SPLIT (60/20/20)
# ═════════════════════════════════════════════════════════════════
print("\n[4/7] Patient-wise stratified split (60/20/20)...")

rng = np.random.default_rng(42)
pm  = df_15.groupby("PatientID")["SepsisLabel"].max().reset_index()
pm.columns = ["PatientID", "Outcome"]

train_p, val_p, test_p = [], [], []
for outcome in pm["Outcome"].unique():
    pids = pm[pm["Outcome"] == outcome]["PatientID"].values.copy()
    rng.shuffle(pids)
    n  = len(pids)
    nt = int(n * 0.60)
    nv = int(n * 0.20)
    train_p.extend(pids[:nt])
    val_p.extend(pids[nt:nt+nv])
    test_p.extend(pids[nt+nv:])

train_df = df_15[df_15["PatientID"].isin(train_p)].copy()
val_df   = df_15[df_15["PatientID"].isin(val_p)].copy()
test_df  = df_15[df_15["PatientID"].isin(test_p)].copy()

for name, split in [("Train",train_df),("Val",val_df),("Test",test_df)]:
    print(f"  {name}: {split['PatientID'].nunique():>5} patients | "
          f"{len(split):>8,} rows | {split['SepsisLabel'].mean():.1%} positive")

# ═════════════════════════════════════════════════════════════════
# STEP 5a — LIGHTGBM WEARABLE STREAM (Stream 1)
# ═════════════════════════════════════════════════════════════════
print("\n[5/7] Training LightGBM wearable stream (Stream 1)...")

# Verify all features exist in the dataframe
missing = [c for c in VITAL_FEAT if c not in df_15.columns]
if missing:
    print(f"  WARNING: Missing VITAL_FEAT columns: {missing}")
    print("  Adding zero-filled columns...")
    for c in missing:
        train_df[c] = val_df[c] = test_df[c] = 0.0

X_tr = train_df[VITAL_FEAT].values
y_tr = train_df["SepsisLabel"].values
X_vl = val_df[VITAL_FEAT].values
y_vl = val_df["SepsisLabel"].values
X_te = test_df[VITAL_FEAT].values
y_te = test_df["SepsisLabel"].values

# Class-weighted training
counts = Counter(y_tr.tolist())
pos_w  = counts[0] / max(counts[1], 1)
w_tr   = np.where(y_tr == 1, pos_w, 1.0)

dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=VITAL_FEAT)
dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)

lgbm_params = {
    "objective":         "binary",
    "metric":            ["auc", "binary_logloss"],
    "boosting_type":     "gbdt",
    "learning_rate":     0.03,
    "max_depth":         7,
    "num_leaves":        63,
    "min_child_samples": 50,
    "feature_fraction":  0.75,
    "bagging_fraction":  0.80,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        0.2,
    "verbose":           -1,
    "random_state":      42,
    "n_jobs":            -1,
}

lgbm_model = lgb.train(
    lgbm_params, dtrain,
    num_boost_round=1000,
    valid_sets=[dval],
    callbacks=[
        lgb.early_stopping(80, verbose=False),
        lgb.log_evaluation(200),
    ],
)

# Validate stored feature names match VITAL_FEAT
stored_names = lgbm_model.feature_name()
assert stored_names == VITAL_FEAT, (
    f"CRITICAL: LightGBM stored {len(stored_names)} names "
    f"but VITAL_FEAT has {len(VITAL_FEAT)}: "
    f"mismatch at positions "
    f"{[i for i,(a,b) in enumerate(zip(stored_names,VITAL_FEAT)) if a!=b]}"
)

# Inject TFT_Score into train/val/test for XGB
train_df["TFT_Score"] = lgbm_model.predict(X_tr)
val_df["TFT_Score"]   = lgbm_model.predict(X_vl)
test_df["TFT_Score"]  = lgbm_model.predict(X_te)

s1_auroc = roc_auc_score(y_te, test_df["TFT_Score"].values)
s1_auprc = average_precision_score(y_te, test_df["TFT_Score"].values)
print(f"  Stream 1 — AUROC: {s1_auroc:.4f} | AUPRC: {s1_auprc:.4f}")

# ═════════════════════════════════════════════════════════════════
# STEP 5b — XGBOOST LAB BOOSTER (Stream 2)
# ═════════════════════════════════════════════════════════════════
print("\n[5/7] Training XGBoost lab booster (Stream 2)...")

# Verify all LAB_BOOSTER_COLS exist (TFT_Score was just injected above)
for c in LAB_BOOSTER_COLS:
    if c not in train_df.columns:
        train_df[c] = val_df[c] = test_df[c] = 0.0

# Gate: only train on rows where at least one lab was actually drawn
tested_cols = [c for c in LAB_BOOSTER_COLS if c.endswith("_Tested")]
has_lab_tr  = train_df[tested_cols].sum(axis=1).values > 0
has_lab_vl  = val_df[tested_cols].sum(axis=1).values   > 0
has_lab_te  = test_df[tested_cols].sum(axis=1).values  > 0

print(f"  Rows with labs — train: {has_lab_tr.sum():,} | "
      f"val: {has_lab_vl.sum():,} | test: {has_lab_te.sum():,}")
print(f"  XGB feature count: {len(LAB_BOOSTER_COLS)}")

X_xtr = train_df.loc[has_lab_tr, LAB_BOOSTER_COLS].values
y_xtr = train_df.loc[has_lab_tr, "SepsisLabel"].values
X_xvl = val_df.loc[has_lab_vl,   LAB_BOOSTER_COLS].values
y_xvl = val_df.loc[has_lab_vl,   "SepsisLabel"].values
X_xte = test_df.loc[has_lab_te,  LAB_BOOSTER_COLS].values
y_xte = test_df.loc[has_lab_te,  "SepsisLabel"].values

# Important: use .values (numpy) NOT DataFrame so xgb_booster.json
# has NO stored feature names. Inference uses column position order.
ratio = float((y_xtr == 0).sum() / max((y_xtr == 1).sum(), 1))

xgb_model = xgb.XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.80,
    colsample_bytree=0.75,
    scale_pos_weight=ratio,
    eval_metric="auc",
    early_stopping_rounds=60,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(
    X_xtr, y_xtr,
    eval_set=[(X_xvl, y_xvl)],
    verbose=False,
)

# Validate XGB feature count
assert xgb_model.get_booster().num_features() == len(LAB_BOOSTER_COLS), (
    f"XGB has {xgb_model.get_booster().num_features()} features "
    f"but LAB_BOOSTER_COLS has {len(LAB_BOOSTER_COLS)}"
)

s2_preds = xgb_model.predict_proba(X_xte)[:, 1]
s2_auroc = roc_auc_score(y_xte, s2_preds)
s2_auprc = average_precision_score(y_xte, s2_preds)
print(f"  Stream 2 — AUROC: {s2_auroc:.4f} | AUPRC: {s2_auprc:.4f}")

# ═════════════════════════════════════════════════════════════════
# STEP 5c — PLATT CALIBRATION (fitted on VALIDATION TFT_Score)
# ═════════════════════════════════════════════════════════════════
print("\n[5/7] Fitting Platt scaler on validation TFT_Score...")

# Platt fitted ONLY on TFT_Score (pure LGBM output on the validation set).
# At inference: calibrated_tft = platt.predict_proba([[tft_score]])[0][1]
# This is used for DISPLAY only — the alert thresholds are set on the raw
# fused score (see fusion below).
platt = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
platt.fit(
    val_df[["TFT_Score"]].values,
    val_df["SepsisLabel"].values,
)
print(f"  Platt fitted on {len(val_df):,} validation rows")

# ═════════════════════════════════════════════════════════════════
# STEP 5d — SCORE FUSION
# ═════════════════════════════════════════════════════════════════
tft_te  = test_df["TFT_Score"].values

# XGB score on full test set (LGBM score for no-lab rows)
xgb_te  = tft_te.copy()
xgb_te[has_lab_te] = xgb_model.predict_proba(
    test_df.loc[has_lab_te, LAB_BOOSTER_COLS].values
)[:, 1]

# Fusion weights
W_LGBM_NOLABS = 1.00
W_LGBM_LABS   = 0.40
W_XGB_LABS    = 0.60

w_lgbm  = np.where(has_lab_te, W_LGBM_LABS, W_LGBM_NOLABS)
w_xgb   = np.where(has_lab_te, W_XGB_LABS,  0.00)
final   = w_lgbm * tft_te + w_xgb * xgb_te

y_true  = test_df["SepsisLabel"].values

# ═════════════════════════════════════════════════════════════════
# STEP 6 — THRESHOLD CALIBRATION
# ═════════════════════════════════════════════════════════════════
# Find thresholds that maximise F2 (recall-weighted) on validation set
# then verify sensitivity/specificity on test set.
print("\n[6/7] Calibrating alert thresholds on validation set...")

tft_vl  = val_df["TFT_Score"].values
xgb_vl  = tft_vl.copy()
xgb_vl[has_lab_vl] = xgb_model.predict_proba(
    val_df.loc[has_lab_vl, LAB_BOOSTER_COLS].values
)[:, 1]

w_l_vl  = np.where(has_lab_vl, W_LGBM_LABS, W_LGBM_NOLABS)
w_x_vl  = np.where(has_lab_vl, W_XGB_LABS,  0.00)
final_vl = w_l_vl * tft_vl + w_x_vl * xgb_vl
y_val   = val_df["SepsisLabel"].values

best_f2, best_thresh_red, best_thresh_amber = 0, 0.55, 0.40

for t_red in np.arange(0.35, 0.75, 0.02):
    for t_amber in np.arange(0.20, t_red - 0.05, 0.02):
        pred = (final_vl >= t_red).astype(int)
        tp = ((pred == 1) & (y_val == 1)).sum()
        fp = ((pred == 1) & (y_val == 0)).sum()
        fn = ((pred == 0) & (y_val == 1)).sum()
        if tp + fp + fn == 0: continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
        if f2 > best_f2:
            best_f2 = f2
            best_thresh_red   = round(t_red, 2)
            best_thresh_amber = round(t_amber, 2)

print(f"  Optimal thresholds — RED: {best_thresh_red}  AMBER: {best_thresh_amber}  (F2={best_f2:.4f})")

# ═════════════════════════════════════════════════════════════════
# STEP 7 — EVALUATION
# ═════════════════════════════════════════════════════════════════
print("\n[7/7] Evaluating on test set...")

THRESH_RED   = best_thresh_red
THRESH_AMBER = best_thresh_amber

auroc = roc_auc_score(y_true, final)
auprc = average_precision_score(y_true, final)
y_pred = (final >= THRESH_RED).astype(int)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp_ = cm.ravel()
sens = tp_ / (tp_ + fn)
spec = tn  / (tn  + fp)
ppv  = tp_ / (tp_ + fp) if (tp_ + fp) > 0 else 0

print("\n" + "="*60)
print("  EVALUATION RESULTS")
print("="*60)
print(f"  Stream 1  AUROC (vitals only) : {s1_auroc:.4f}")
print(f"  Stream 2  AUROC (lab booster) : {s2_auroc:.4f}")
print(f"  Fusion    AUROC               : {auroc:.4f}  (target >0.82)")
print(f"  Fusion    AUPRC               : {auprc:.4f}  (target >0.40)")
print(f"  Sensitivity  @ RED threshold  : {sens:.4f}  (target >0.80)")
print(f"  Specificity  @ RED threshold  : {spec:.4f}  (target >0.80)")
print(f"  PPV (precision)               : {ppv:.4f}")

tiers = np.where(final >= THRESH_RED, "RED",
        np.where(final >= THRESH_AMBER, "AMBER", "GREEN"))
print(f"\n  Alert tier distribution:")
for tier in ["RED", "AMBER", "GREEN"]:
    mask = tiers == tier
    tp_r = y_true[mask].mean() if mask.sum() > 0 else 0
    print(f"    {tier:<6}: {mask.sum():>7,} rows | {tp_r:.1%} truly septic")

print(f"\n  Validation gates:")
print(f"    AUROC > 0.82     : {'✓ PASS' if auroc > 0.82 else '✗ FAIL'}")
print(f"    AUPRC > 0.40     : {'✓ PASS' if auprc > 0.40 else '✗ FAIL'}")
print(f"    Sensitivity>0.80 : {'✓ PASS' if sens  > 0.80 else '✗ FAIL'}")
print(f"    Specificity>0.80 : {'✓ PASS' if spec  > 0.80 else '✗ FAIL'}")

# ── Calibration check on Platt-scaled TFT_Score ──────────────────
print("\nCalibration (Platt-scaled TFT_Score on test set):")
tft_cal = platt.predict_proba(test_df[["TFT_Score"]].values)[:, 1]
prob_true, prob_pred = calibration_curve(y_true, tft_cal, n_bins=10)
max_cal_err = np.max(np.abs(prob_true - prob_pred))
print(f"  Max calibration error: {max_cal_err:.4f}  (good if <0.08)")
for pt, pp in zip(prob_true, prob_pred):
    bar = "▌" * int(pp * 30)
    print(f"  Pred {pp:.2f} → True {pt:.2f}  {bar}")

# ── Plots ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# ROC
fpr, tpr, _ = roc_curve(y_true, final)
axes[0].plot(fpr, tpr, color="#2563EB", lw=2, label=f"Fusion (AUROC={auroc:.3f})")
fpr1, tpr1, _ = roc_curve(y_te, tft_te)
axes[0].plot(fpr1, tpr1, "--", color="#9CA3AF", lw=1.5, label=f"Vitals ({s1_auroc:.3f})")
axes[0].plot([0,1],[0,1], ":", color="gray", lw=1)
axes[0].set(title="ROC curve", xlabel="FPR", ylabel="TPR")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# PR curve
prec, rec, _ = precision_recall_curve(y_true, final)
axes[1].plot(rec, prec, color="#0D6B6B", lw=2, label=f"Fusion (AUPRC={auprc:.3f})")
axes[1].axhline(y=y_true.mean(), color="gray", linestyle="--", lw=1, label=f"Baseline")
axes[1].set(title="Precision-recall", xlabel="Recall", ylabel="Precision")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

# Calibration
axes[2].plot(prob_pred, prob_true, "o-", color="#2563EB", lw=2, label="Platt(TFT_Score)")
axes[2].plot([0,1],[0,1], "--", color="gray", label="Perfect")
axes[2].set(title="Calibration", xlabel="Predicted probability", ylabel="True fraction")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

# Alert tier PPV
tier_ppvs = []
for tier in ["GREEN","AMBER","RED"]:
    mask = tiers == tier
    ppv_t = y_true[mask].mean() if mask.sum() > 0 else 0
    tier_ppvs.append(ppv_t)
axes[3].bar(["GREEN","AMBER","RED"], tier_ppvs,
            color=["#38A169","#DD6B20","#E53E3E"], alpha=0.8)
axes[3].set(title="PPV per alert tier", ylabel="Fraction truly septic", ylim=[0,1])
axes[3].grid(alpha=0.3, axis="y")

plt.suptitle("Coimbatore Sepsis AI — Evaluation Results", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("/content/evaluation_results.png", bbox_inches="tight", dpi=150)
plt.show()
print("\n✓ Plots saved: /content/evaluation_results.png")

# ── SHAP ─────────────────────────────────────────────────────────
print("\nComputing SHAP values (top 20 wearable features)...")
sample_idx = np.random.choice(len(X_te), min(500, len(X_te)), replace=False)
explainer  = shap.TreeExplainer(lgbm_model)
shap_vals  = explainer.shap_values(X_te[sample_idx])
sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_te[sample_idx], feature_names=VITAL_FEAT,
                  show=False, max_display=20)
plt.title("SHAP — Top 20 wearable features (Stream 1)")
plt.tight_layout()
plt.savefig("/content/shap_summary.png", bbox_inches="tight", dpi=150)
plt.show()
print("✓ SHAP saved: /content/shap_summary.png")

# ═════════════════════════════════════════════════════════════════
# SAVE ALL 5 ARTIFACTS
# ═════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SAVING ARTIFACTS")
print("="*60)

# 1. LightGBM model
lgbm_model.save_model("/content/lgbm_stream1.txt")
print("✓ lgbm_stream1.txt")

# 2. XGBoost model
xgb_model.save_model("/content/xgb_booster.json")
print("✓ xgb_booster.json")

# 3. Platt scaler
joblib.dump(platt, "/content/platt_scaler.pkl")
print("✓ platt_scaler.pkl")

# 4. Feature registry (single source of truth for column order)
feature_registry = {
    "lgbm_vital_features":  VITAL_FEAT,
    "xgb_lab_booster_cols": LAB_BOOSTER_COLS,
}
with open("/content/feature_registry.json", "w") as f:
    json.dump(feature_registry, f, indent=2)
print(f"✓ feature_registry.json  (LGBM={len(VITAL_FEAT)}, XGB={len(LAB_BOOSTER_COLS)})")

# 5. Inference config (weights + thresholds so inference matches training exactly)
inference_config = {
    "ensemble": {
        "no_labs":  {"lgbm": W_LGBM_NOLABS, "xgb": 0.00},
        "has_labs": {"lgbm": W_LGBM_LABS,   "xgb": W_XGB_LABS},
    },
    "alert_thresholds": {
        "red":   THRESH_RED,
        "amber": THRESH_AMBER,
    },
    "evaluation": {
        "fusion_auroc":   round(float(auroc),  4),
        "fusion_auprc":   round(float(auprc),  4),
        "sensitivity":    round(float(sens),   4),
        "specificity":    round(float(spec),   4),
        "s1_lgbm_auroc":  round(float(s1_auroc), 4),
        "s2_xgb_auroc":   round(float(s2_auroc), 4),
    },
    "xgb_trained_on_has_lab_only": True,
    "platt_input_feature":         "TFT_Score",
    "platt_used_for":              "display_calibration_only",
}
with open("/content/inference_config.json", "w") as f:
    json.dump(inference_config, f, indent=2)
print("✓ inference_config.json")

# ── Final validation ──────────────────────────────────────────────
print("\n── Post-save validation ────────────────────────────────────")
_m = lgb.Booster(model_file="/content/lgbm_stream1.txt")
assert _m.feature_name() == VITAL_FEAT,           "✗ LightGBM feature name mismatch"
print(f"  ✓ LightGBM feature names validated ({len(VITAL_FEAT)} features)")

_x = xgb.Booster(); _x.load_model("/content/xgb_booster.json")
assert _x.num_features() == len(LAB_BOOSTER_COLS), "✗ XGBoost feature count mismatch"
print(f"  ✓ XGBoost feature count validated  ({len(LAB_BOOSTER_COLS)} features)")

_p = joblib.load("/content/platt_scaler.pkl")
assert hasattr(_p, "predict_proba"),               "✗ Platt scaler broken"
print(f"  ✓ Platt scaler validated")

print("\n" + "="*60)
print("  ALL 5 ARTIFACTS SAVED TO /content/")
print("  Copy to backend/ before starting the server.")
print("="*60)
print("\n  lgbm_stream1.txt")
print("  xgb_booster.json")
print("  platt_scaler.pkl")
print("  feature_registry.json")
print("  inference_config.json")
