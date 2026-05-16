"""
live_test.py — End-to-end API wiring test
Run from the backend directory with the venv's Python.
Tests every endpoint and prints a pass/fail report.
"""

import sys
import json
import requests

BASE = "http://localhost:8000"
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
results = []

def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    results.append((name, ok, detail))
    print(f"  {tag}  {name}")
    if detail:
        for line in detail.split("\n"):
            print(f"       {line}")

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

# ── 1. Health ────────────────────────────────────────────────
section("1. GET /api/health")
try:
    r = requests.get(f"{BASE}/api/health", timeout=10)
    d = r.json()
    check("HTTP 200", r.status_code == 200, f"status={r.status_code}")
    check("status = ok or degraded", d.get("status") in ("ok","degraded"), f"status={d.get('status')}")
    check("lgbm_loaded = True",  d.get("lgbm_loaded")  is True)
    check("xgb_loaded = True",   d.get("xgb_loaded")   is True)
    check("platt_loaded = True",  d.get("platt_loaded")  is True)
    check("registry_loaded = True", d.get("registry_loaded") is True)
    check("fused_platt_loaded",  d.get("fused_platt_loaded") is True,
          "(False means fused calibration disabled)")
    print(f"       Full response: {json.dumps(d, indent=6)[:400]}")
except Exception as e:
    check("GET /api/health", False, str(e))

# ── 2. Patients list ─────────────────────────────────────────
section("2. GET /api/patients")
patient_ids = []
try:
    r = requests.get(f"{BASE}/api/patients", timeout=10)
    d = r.json()
    check("HTTP 200", r.status_code == 200)
    patients = d.get("patients", [])
    check("patients list non-empty", len(patients) > 0, f"count={len(patients)}")
    patient_ids = [p["patient_id"] for p in patients]
    for p in patients:
        has_required = all(k in p for k in ["patient_id","label","total_readings","interval_minutes","has_labs"])
        check(f"  {p['patient_id']} shape ok", has_required,
              f"total={p.get('total_readings')} interval={p.get('interval_minutes')}min labs={p.get('has_labs')}")
except Exception as e:
    check("GET /api/patients", False, str(e))

# ── 3. Features ──────────────────────────────────────────────
section("3. GET /api/features")
try:
    r = requests.get(f"{BASE}/api/features", timeout=10)
    d = r.json()
    check("HTTP 200", r.status_code == 200)
    lgbm_n = d.get("lgbm_count", 0)
    xgb_n  = d.get("xgb_count",  0)
    check("LGBM feature count = 36", lgbm_n == 36, f"got {lgbm_n}")
    check("XGB  feature count = 51", xgb_n  == 51, f"got {xgb_n}")
    check("source = registry", d.get("source") == "registry", f"source={d.get('source')}")
except Exception as e:
    check("GET /api/features", False, str(e))

# ── 4. POST /api/predict (no labs) ──────────────────────────
section("4. POST /api/predict  [vitals-only, no labs]")
PREDICT_BODY_NOLABS = {
    "vitals": {
        "hr": 105.0, "map": 70.0, "resp": 22.0, "temp": 38.6,
        "o2sat": 95.0, "urineOutput": 0.4, "gcs": 14, "systolicBp": 100.0
    },
    "labs": {
        "lactate":    {"value": 0.0, "performed": False},
        "pct":        {"value": 0.0, "performed": False},
        "wbc":        {"value": 0.0, "performed": False},
        "platelets":  {"value": 0.0, "performed": False},
        "creatinine": {"value": 0.0, "performed": False},
        "bilirubin":  {"value": 0.0, "performed": False},
        "crp":        {"value": 0.0, "performed": False},
        "dengueNS1":  {"value": 0,   "performed": False},
        "malariaRDT": {"value": 0,   "performed": False},
    },
    "demographics": {
        "age": 55, "gender": "Male", "bmi": 27.0,
        "diabetes": True, "ckd": False, "cirrhosis": False,
        "malignancy": False, "immunosuppression": False,
        "priorAntibiotics": False, "referredFromOutside": True,
        "gramNegativeRisk": False
    },
    "intervalSeconds": 0
}
try:
    r = requests.post(f"{BASE}/api/predict", json=PREDICT_BODY_NOLABS, timeout=15)
    check("HTTP 200", r.status_code == 200, f"status={r.status_code} body={r.text[:200]}")
    if r.status_code == 200:
        d = r.json()
        check("aiScore present",          "aiScore" in d, f"aiScore={d.get('aiScore')}")
        check("aiScore in [0,1]",         0 <= (d.get("aiScore") or 0) <= 1, f"aiScore={d.get('aiScore')}")
        check("alertLevel present",       d.get("alertLevel") in ("none","warning","critical","sensor_error"))
        check("qsofaScore present",       "qsofaScore" in d, f"qsofaScore={d.get('qsofaScore')}")
        check("qsofaComponents present",  "qsofaComponents" in d)
        check("riskFactors list",         isinstance(d.get("riskFactors"), list))
        check("deltaInfo present",        "deltaInfo" in d)
        check("confidenceScore present",  "confidenceScore" in d)
        check("dataQualityWarnings list", isinstance(d.get("dataQualityWarnings"), list))
        check("monitoringUrgencyWeights", len(d.get("monitoringUrgencyWeights", [])) == 6)
        check("featureSummary present",   "featureSummary" in d)
        check("featureSummary.crt",       "crt" in d.get("featureSummary", {}))
        check("featureSummary.hasLabs",   "hasLabs" in d.get("featureSummary", {}))
        check("modelScores.lgbm_tft",     d.get("modelScores",{}).get("lgbm_tft") is not None)
        # No labs → XGB should NOT run
        check("xgbActive=False (no labs)", d.get("featureSummary",{}).get("xgbActive") == False,
              f"xgbActive={d.get('featureSummary',{}).get('xgbActive')}")
        print(f"\n       aiScore={d['aiScore']}  alert={d['alertLevel']}  qSOFA={d['qsofaScore']}")
        print(f"       confidence={d.get('confidenceScore')}  lgbm_tft={d['modelScores'].get('lgbm_tft')}")
except Exception as e:
    check("POST /api/predict [no labs]", False, str(e))

# ── 5. POST /api/predict (with labs) ────────────────────────
section("5. POST /api/predict  [with 7 labs, XGB should activate]")
PREDICT_BODY_LABS = {
    **PREDICT_BODY_NOLABS,
    "labs": {
        "lactate":    {"value": 2.8, "performed": True},
        "pct":        {"value": 5.2, "performed": True},
        "wbc":        {"value": 14.5,"performed": True},
        "platelets":  {"value": 150, "performed": True},
        "creatinine": {"value": 1.8, "performed": True},
        "bilirubin":  {"value": 1.1, "performed": True},
        "crp":        {"value": 85,  "performed": True},
        "dengueNS1":  {"value": 0,   "performed": False},
        "malariaRDT": {"value": 0,   "performed": False},
    }
}
try:
    r = requests.post(f"{BASE}/api/predict", json=PREDICT_BODY_LABS, timeout=15)
    check("HTTP 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        d = r.json()
        check("xgbActive=True (7 labs)",  d.get("featureSummary",{}).get("xgbActive") == True,
              f"xgbActive={d.get('featureSummary',{}).get('xgbActive')}")
        check("xgb_lab score present",    d.get("modelScores",{}).get("xgb_lab") is not None,
              f"xgb_lab={d.get('modelScores',{}).get('xgb_lab')}")
        check("mimicInfo present",        "mimicInfo" in d)
        check("aiScore > 0.3 (high risk patient)", (d.get("aiScore") or 0) > 0.3,
              f"aiScore={d.get('aiScore')}")
        print(f"\n       aiScore={d['aiScore']}  alert={d['alertLevel']}  qSOFA={d['qsofaScore']}")
        print(f"       xgb_lab={d['modelScores'].get('xgb_lab')}  fused_raw={d['modelScores'].get('fused_raw')}")
        print(f"       warnings: {d.get('dataQualityWarnings')}")
except Exception as e:
    check("POST /api/predict [with labs]", False, str(e))

# ── 6. Validation: MAP > systolicBp should → 422 ────────────
section("6. POST /api/predict  [invalid: MAP > systolicBp → should get 422]")
BAD_BODY = {**PREDICT_BODY_NOLABS}
BAD_BODY["vitals"] = {**PREDICT_BODY_NOLABS["vitals"], "map": 150.0, "systolicBp": 80.0}
try:
    r = requests.post(f"{BASE}/api/predict", json=BAD_BODY, timeout=10)
    check("HTTP 422 for MAP > SBP", r.status_code == 422,
          f"got {r.status_code}: {r.text[:150]}")
except Exception as e:
    check("Validation reject", False, str(e))

# ── 7. Session pipeline ──────────────────────────────────────
section("7. Session pipeline:  start → tick × 3 → status → stop")
session_id = None
test_patient = "DEMO_SEPSIS_FULLLABS" if "DEMO_SEPSIS_FULLLABS" in patient_ids else (patient_ids[0] if patient_ids else None)

if not test_patient:
    check("Session test", False, "No patient files available")
else:
    # 7a. start
    try:
        r = requests.post(f"{BASE}/api/session/start",
                          json={"patient_id": test_patient}, timeout=15)
        check(f"start HTTP 200 ({test_patient})", r.status_code == 200,
              f"status={r.status_code} body={r.text[:200]}")
        if r.status_code == 200:
            d = r.json()
            session_id = d.get("_meta", {}).get("session_id")
            check("_meta.session_id present", session_id is not None)
            check("_meta.reading_index = 0",  d.get("_meta",{}).get("reading_index") == 0,
                  f"reading_index={d.get('_meta',{}).get('reading_index')}")
            check("_meta.total_readings > 0",  d.get("_meta",{}).get("total_readings",0) > 0,
                  f"total={d.get('_meta',{}).get('total_readings')}")
            check("aiScore present in start",  "aiScore" in d, f"aiScore={d.get('aiScore')}")
            check("vitals present in start",   "vitals" in d)
            check("labs present in start",     "labs" in d)
            check("demographics present",      "demographics" in d)
            check("riskFactors list",          isinstance(d.get("riskFactors"), list))
            print(f"\n       reading 0/{d['_meta']['total_readings']}  "
                  f"aiScore={d.get('aiScore')}  alert={d.get('alertLevel')}")
            print(f"       vitals: HR={d['vitals']['hr']}  MAP={d['vitals']['map']}  "
                  f"Temp={d['vitals']['temp']}  SBP={d['vitals']['systolicBp']}")
    except Exception as e:
        check("session/start", False, str(e))

    # 7b. tick × 3
    for tick_n in range(1, 4):
        try:
            r = requests.post(f"{BASE}/api/session/tick", timeout=15)
            check(f"tick {tick_n} HTTP 200", r.status_code == 200,
                  f"status={r.status_code}")
            if r.status_code == 200:
                d = r.json()
                idx = d.get("_meta",{}).get("reading_index", "?")
                ai  = d.get("aiScore")
                alv = d.get("alertLevel")
                hr  = d.get("vitals",{}).get("hr","?")
                print(f"       reading {idx}  aiScore={ai}  alert={alv}  HR={hr}")
                check(f"tick {tick_n} has aiScore", "aiScore" in d)
                check(f"tick {tick_n} has vitals",  "vitals" in d)
        except Exception as e:
            check(f"tick {tick_n}", False, str(e))

    # 7c. session/status
    try:
        r = requests.get(f"{BASE}/api/session/status", timeout=10)
        check("session/status HTTP 200", r.status_code == 200)
        d = r.json()
        check("active_sessions >= 1", d.get("active_sessions", 0) >= 1,
              f"active_sessions={d.get('active_sessions')}")
    except Exception as e:
        check("session/status", False, str(e))

    # 7d. stop
    try:
        r = requests.post(f"{BASE}/api/session/stop", timeout=10)
        check("session/stop HTTP 200", r.status_code == 200)
        check("stopped=True",         r.json().get("stopped") is True)
    except Exception as e:
        check("session/stop", False, str(e))

# ── 8. Invalid patient ID ────────────────────────────────────
section("8. session/start [invalid patient_id → should get 422]")
try:
    r = requests.post(f"{BASE}/api/session/start",
                      json={"patient_id": "../../etc/passwd"}, timeout=10)
    check("Path traversal rejected (422)", r.status_code == 422,
          f"got {r.status_code}: {r.text[:150]}")
except Exception as e:
    check("Path traversal test", False, str(e))

# ── Summary ──────────────────────────────────────────────────
section("SUMMARY")
passed  = sum(1 for _,ok,_ in results if ok)
failed  = sum(1 for _,ok,_ in results if not ok)
print(f"\n  {PASS} Passed : {passed}")
print(f"  {FAIL} Failed : {failed}")
print(f"  Total  : {passed + failed}\n")
if failed > 0:
    print("  Failed checks:")
    for name, ok, detail in results:
        if not ok:
            print(f"    {FAIL} {name}  {detail}")

sys.exit(0 if failed == 0 else 1)
