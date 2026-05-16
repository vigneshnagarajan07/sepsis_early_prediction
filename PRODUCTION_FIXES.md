# Sepsis AI — Production Bug Fixes

All 17 bugs from the audit have been resolved.  Below is the complete
fix log with the file, the problem, and what changed.

---

## Backend

### features.py

| # | Bug | Fix |
|---|-----|-----|
| 1 | `_lab_scenario` coverage gap — 1 or 2 non-CBC labs returned `"full"` instead of `"partial_cbc"` | Rewrote tier logic: 0→no_labs, 1-2→partial_cbc, 3-4→partial_full, 5+→full |
| 2 | qSOFA SBP threshold was ambiguous — must be `≤ 100` (Seymour 2016), not `< 90` (shock) | Added explicit comment confirming `int(systolic_bp <= 100)` with test coverage |
| 3 | NaN propagation — unperformed labs with `None` value could produce `nan` array entries | Added `_NAN_FALLBACKS` dict; all XGB_FEATURES are scrubbed after `feat_dict` assembly |

### models.py

| # | Bug | Fix |
|---|-----|-----|
| 4 | Feature vector mismatch — XGB DMatrix built from raw ndarray with no column names; XGBoost could silently reorder | `_xgb_predict` now builds a named pandas DataFrame → `xgb.DMatrix(df)` enforces column order |
| 5 | Health endpoint stale flags — `_xgb_ok`/`_lgbm_ok` module-level booleans could be stale | `health()` now derives status from `_xgb_model is not None` at call time |
| 6 | LightGBM NaN output unguarded — degenerate input could return `nan` probability | Added `np.isnan` guard in `_lgbm_predict`; returns `0.5` (max uncertainty) instead |
| 7 | Ensemble NaN propagation — if both models returned `nan`, `ai_score` was `nan` | Added `np.isnan(ai_score)` guard before clipping; falls back to `0.5` |

### main.py

| # | Bug | Fix |
|---|-----|-----|
| 8 | Monolithic exception handler — all errors returned HTTP 500 including "no models" | Differentiated: `RuntimeError→503`, `ValueError→422`, other→500 with traceback log |
| 9 | CORS wildcard + credentials conflict (browser spec violation) | `allow_credentials=True` only when `ALLOWED_ORIGINS` env var is set; wildcard disables credentials |
| 10 | MAP > SBP cross-field validation missing | Added `@model_validator` on `Vitals`; raises 422 if MAP > systolicBp |

### requirements.txt

| # | Bug | Fix |
|---|-----|-----|
| 11 | LightGBM unpinned — `.txt` model format is version-sensitive | Pinned to `>=4.3.0,<4.4.0` to match training environment |
| 12 | pandas missing — needed by `_xgb_predict` for named DMatrix | Added `pandas>=2.0.0,<3.0.0` |

### test_cases.py

| # | Bug | Fix |
|---|-----|-----|
| 13 | `covidPrevalence` field sent but not in Demographics Pydantic model → silent 422 | Removed from all three test payloads |
| 14 | Missing demographic fields (`cirrhosis`, `malignancy`, `immunosuppression`) in test payloads | Added to all three cases |
| 15 | No offline unit tests — running tests required live server + model files | Added `--unit` mode: 17 assertions covering qSOFA threshold, lab scenario tiers, NaN guard, vector lengths, delta baseline index |

---

## Frontend (src/App.tsx)

| # | Bug | Fix |
|---|-----|-----|
| 16 | `runAnalysis` had no in-flight guard — simulation mode stacked concurrent requests every 5 s | Added `analysisInFlight` ref; skips call if one is already running |
| 17 | API errors swallowed by `console.error` — clinician saw a blank UI with no feedback | Added `analysisError` state; renders a visible red banner below the vitals section |
| 18 | Simulation auto-analysis `useEffect` deps included `[vitals, labs, demographics]` — fired on every keystroke | Fixed deps to `[isSimulating, runAnalysis]`; interval callback calls `runAnalysis()` directly |
| 19 | D3 `AttentionHeatmap`, `DetailedChart`, `Sparkline` — no cleanup return from `useEffect` → double SVG elements in React 19 StrictMode | Added `return () => { d3.select(svgRef.current).selectAll("*").remove(); }` to all three components |
| 20 | `useCallback` missing on `runAnalysis` — new function reference on every render broke `useEffect` deps | Wrapped `runAnalysis` in `useCallback` |

---

## How to start (development)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev          # Express proxy on :3000 → FastAPI on :8000

# Unit tests (no server needed)
cd backend
python test_cases.py --unit

# Integration tests (server must be running)
python test_cases.py
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | same dir as `main.py` | Path to `xgb_booster.json` and `lgbm_stream1.txt` |
| `XGB_FEATURE_ORDER_CONFIRMED` | `0` | Set to `1` after verifying XGB column order matches training script |
| `ALLOWED_ORIGINS` | `""` (wildcard) | Comma-separated CORS origins for production deployment |
| `FASTAPI_URL` | `http://localhost:8000` | Target for Express proxy in `server.ts` |
| `VITE_API_URL` | `http://localhost:8000` | Target for Vite dev proxy in `vite.config.ts` |

