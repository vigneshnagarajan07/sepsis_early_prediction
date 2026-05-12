# ═══════════════════════════════════════════════════════════════════
# PATCH CELL — Add this IMMEDIATELY after the existing training cell
# Saves all artifacts needed by the inference stack
# ═══════════════════════════════════════════════════════════════════
import joblib, json

# ── Save Platt scaler (was fitted but never saved — critical bug) ──
joblib.dump(platt, "/content/platt_scaler.pkl")
print("✓ platt_scaler.pkl saved")

# ── Save feature column lists ──────────────────────────────────────
# These are the ground-truth lists the models were ACTUALLY trained on.
# Inference code must use exactly these, in exactly this order.
feature_registry = {
    "lgbm_vital_features":  VITAL_FEAT,
    "xgb_lab_booster_cols": LAB_BOOSTER_COLS,
}
with open("/content/feature_registry.json", "w") as f:
    json.dump(feature_registry, f, indent=2)
print(f"✓ feature_registry.json saved")
print(f"  LightGBM features  : {len(VITAL_FEAT)}")
print(f"  XGBoost features   : {len(LAB_BOOSTER_COLS)}")

# ── Save ensemble weights & thresholds used at evaluation ─────────
inference_config = {
    "ensemble": {
        "no_labs": {"lgbm": 1.00, "xgb": 0.00},
        "has_labs": {"lgbm": 0.40, "xgb": 0.60},
    },
    "alert_thresholds": {
        "red":   0.55,   # THRESH_RED from training notebook
        "amber": 0.40,   # THRESH_AMBER from training notebook
    },
    "platt_features": ["TFT_Score"],   # Platt was fitted on TFT_Score only
    "xgb_trained_on_has_lab_only": True,
}
with open("/content/inference_config.json", "w") as f:
    json.dump(inference_config, f, indent=2)
print("✓ inference_config.json saved")

# ── Validate LightGBM stored feature names match what we expect ───
stored_names = lgbm_model.feature_name()
assert stored_names == VITAL_FEAT, (
    f"LightGBM stored {len(stored_names)} feature names "
    f"but VITAL_FEAT has {len(VITAL_FEAT)} — mismatch!"
)
print(f"✓ LightGBM feature names validated ({len(stored_names)} features)")

# ── Quick XGBoost feature count check ─────────────────────────────
xgb_n = xgb_model.get_booster().num_features()
assert xgb_n == len(LAB_BOOSTER_COLS), (
    f"XGBoost expects {xgb_n} features but LAB_BOOSTER_COLS has "
    f"{len(LAB_BOOSTER_COLS)} — column order will be wrong!"
)
print(f"✓ XGBoost feature count validated ({xgb_n} features)")
print(f"\n✓ All 4 artifacts saved. Copy to backend/:")
print(f"   lgbm_stream1.txt, xgb_booster.json,")
print(f"   platt_scaler.pkl, feature_registry.json, inference_config.json")
