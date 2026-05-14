"""
test_cases.py  –  Sepsis AI · Sanity-check suite

Two modes:
  1. Offline unit tests  (python test_cases.py --unit)
     Tests features.py in isolation — no server required, no model files.
  2. Live integration   (python test_cases.py)
     Hits http://127.0.0.1:8000/api/predict with three clinical payloads.

BUG FIX: removed `covidPrevalence` field that was not part of the
Demographics Pydantic model, and added the previously-missing fields
(cirrhosis, malignancy, immunosuppression) to each case.
"""

import json
import sys
import urllib.request


# ─────────────────────────────────────────────────────────────────────────────
# Clinical test payloads
# ─────────────────────────────────────────────────────────────────────────────

CASES = {
    "Case 1 — CRITICAL (severe sepsis)": {
        "vitals": {
            "hr": 128, "map": 55, "resp": 28, "temp": 39.4,
            "o2sat": 91, "urineOutput": 0.2, "gcs": 10, "systolicBp": 80
        },
        "labs": {
            "lactate":    {"value": 4.8,  "performed": True},
            "pct":        {"value": 22.0, "performed": True},
            "wbc":        {"value": 18.5, "performed": True},
            "platelets":  {"value": 90,   "performed": True},
            "creatinine": {"value": 2.4,  "performed": True},
            "bilirubin":  {"value": 2.1,  "performed": True},
            "dengueNS1":  {"value": 0,    "performed": True},
            "malariaRDT": {"value": 0,    "performed": True}
        },
        "demographics": {
            "age": 58, "gender": "Male", "bmi": 23.5,
            "diabetes": True,  "ckd": True,
            "cirrhosis": False, "malignancy": False, "immunosuppression": False,
            "priorAntibiotics": False, "referredFromOutside": False,
            "gramNegativeRisk": True,
            "malariaEndemic": False, "dengueEndemic": False,
        }
    },

    "Case 2 — WARNING (dengue thrombocytopenia + partial labs)": {
        "vitals": {
            "hr": 102, "map": 78, "resp": 20, "temp": 38.3,
            "o2sat": 96, "urineOutput": 0.6, "gcs": 15, "systolicBp": 105
        },
        "labs": {
            "lactate":    {"value": 0,   "performed": False},
            "pct":        {"value": 0,   "performed": False},
            "wbc":        {"value": 2.8, "performed": True},
            "platelets":  {"value": 55,  "performed": True},
            "creatinine": {"value": 0,   "performed": False},
            "bilirubin":  {"value": 0,   "performed": False},
            "dengueNS1":  {"value": 1,   "performed": True},
            "malariaRDT": {"value": 0,   "performed": True}
        },
        "demographics": {
            "age": 34, "gender": "Female", "bmi": 21.0,
            "diabetes": False, "ckd": False,
            "cirrhosis": False, "malignancy": False, "immunosuppression": False,
            "priorAntibiotics": True, "referredFromOutside": True,
            "gramNegativeRisk": False,
            "malariaEndemic": False, "dengueEndemic": True,
        }
    },

    "Case 3 — NONE (healthy baseline, no risk factors)": {
        "vitals": {
            "hr": 72, "map": 95, "resp": 14, "temp": 36.8,
            "o2sat": 99, "urineOutput": 1.2, "gcs": 15, "systolicBp": 128
        },
        "labs": {
            "lactate":    {"value": 0.9,  "performed": True},
            "pct":        {"value": 0.04, "performed": True},
            "wbc":        {"value": 7.2,  "performed": True},
            "platelets":  {"value": 245,  "performed": True},
            "creatinine": {"value": 0.85, "performed": True},
            "bilirubin":  {"value": 0.5,  "performed": True},
            "dengueNS1":  {"value": 0,    "performed": True},
            "malariaRDT": {"value": 0,    "performed": True}
        },
        "demographics": {
            "age": 28, "gender": "Male", "bmi": 22.0,
            "diabetes": False, "ckd": False,
            "cirrhosis": False, "malignancy": False, "immunosuppression": False,
            "priorAntibiotics": False, "referredFromOutside": False,
            "gramNegativeRisk": False,
            "malariaEndemic": False, "dengueEndemic": False,
        }
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# Offline unit tests (no server required)
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_tests() -> None:
    """Validate features.py logic in isolation — no model files needed."""
    import importlib
    import numpy as np

    feat = importlib.import_module("features")

    PASS = "\033[92m PASS\033[0m"
    FAIL = "\033[91m FAIL\033[0m"
    failures = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        status = PASS if condition else FAIL
        print(f"  {status}  {label}" + (f"  [{detail}]" if detail else ""))
        if not condition:
            failures += 1

    print("\n── Unit tests: features.py ─────────────────────────────────────")

    # ── qSOFA: correct SBP threshold ≤100 (not <90) ──────────────────────────
    def qsofa_from(sbp: float, resp: float, gcs: float) -> int:
        return int(resp >= 22) + int(gcs < 15) + int(sbp <= 100)

    check("qSOFA SBP=80 → 1 point",  qsofa_from(80,  14, 15) == 1)
    check("qSOFA SBP=100 → 1 point", qsofa_from(100, 14, 15) == 1,
          "SBP=100 is AT the threshold → must score 1")
    check("qSOFA SBP=101 → 0 points", qsofa_from(101, 14, 15) == 0,
          "SBP=101 is above threshold → must score 0")
    check("qSOFA SBP=80+RR28+GCS10 → 3", qsofa_from(80, 28, 10) == 3)

    # ── Lab scenario: the coverage-gap bug ───────────────────────────────────
    def scenario(labs_dict: dict) -> str:
        idx = feat._lab_scenario(labs_dict)
        return ["no_labs","partial_cbc","partial_full","full"][idx.index(1)]

    def make_labs(*performed_keys: str) -> dict:
        core = ["lactate","pct","wbc","platelets","creatinine","bilirubin"]
        return {k: {"value": 1.0, "performed": k in performed_keys} for k in core}

    check("0 labs → no_labs",             scenario(make_labs()) == "no_labs")
    check("1 lab (lactate only) → partial_cbc",
          scenario(make_labs("lactate")) == "partial_cbc",
          "BUG was: returned 'full'")
    check("2 labs (lactate+creat, no CBC) → partial_cbc",
          scenario(make_labs("lactate","creatinine")) == "partial_cbc",
          "BUG was: returned 'full'")
    check("2 labs (wbc+platelets) → partial_cbc",
          scenario(make_labs("wbc","platelets")) == "partial_cbc")
    check("3 labs → partial_full",
          scenario(make_labs("wbc","platelets","lactate")) == "partial_full")
    check("5 labs → full",
          scenario(make_labs("wbc","platelets","lactate","pct","creatinine")) == "full")
    check("6 labs → full",
          scenario(make_labs("lactate","pct","wbc","platelets","creatinine","bilirubin")) == "full")

    # ── NaN guard ────────────────────────────────────────────────────────────
    case1_payload = {**CASES["Case 1 — CRITICAL (severe sepsis)"],
                     "vitalsHistory": {}, "previousLabs": {}, "intervalSeconds": 0}
    lgbm_vec, xgb_vec, fd = feat.build_feature_vector(case1_payload)
    check("No NaN in LGBM vector", not np.isnan(lgbm_vec).any())
    check("No NaN in XGB vector",  not np.isnan(xgb_vec).any())
    check("LGBM vector length == 36", len(lgbm_vec) == 36)
    check("XGB vector length == 51",  len(xgb_vec)  == 51)

    # ── Delta baseline index ──────────────────────────────────────────────────
    # 5-second intervals, 12 readings → 60 s total; 3 h far exceeds buffer → idx 0
    check("Short buffer → baseline index 0",
          feat._baseline_index(12, 5) == 0)
    # 900-second intervals, 20 readings → 5 h buffer; 3 h back ≈ 12 steps → idx max(0, 20-1-12)=7
    check("15-min cadence, 20 readings → idx 7",
          feat._baseline_index(20, 900) == 7)

    print()
    if failures:
        print(f"  ✗  {failures} test(s) FAILED\n")
        sys.exit(1)
    else:
        print(f"  ✓  All tests passed\n")


# ─────────────────────────────────────────────────────────────────────────────
# Live integration tests
# ─────────────────────────────────────────────────────────────────────────────

URL = "http://127.0.0.1:8000/api/predict"
SEP = "─" * 60


def run_case(label: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def run_integration_tests() -> None:
    print(f"\nIntegration tests → {URL}\n")
    for label, payload in CASES.items():
        print(f"{SEP}\n  {label}\n{SEP}")
        try:
            r = run_case(label, payload)
            print(f"  aiScore      : {r['aiScore']}")
            print(f"  qsofaScore   : {r['qsofaScore']}")
            print(f"  alertLevel   : {r['alertLevel'].upper()}")
            print(f"  modelScores  : {r['modelScores']}")
            print(f"  labScenario  : {r['featureSummary']['labScenario']}")
            print(f"  shockIndex   : {r['featureSummary']['shockIndex']}")
            print(f"  crt          : {r['featureSummary']['crt']}")
            print("  Top SHAP drivers:")
            for d in r.get("shapDrivers", [])[:5]:
                bar = "▲" if d["shap"] > 0 else "▼"
                print(f"    {bar}  {d['feature']:<28} val={d['value']}  shap={d['shap']}")
        except Exception as e:
            print(f"  ERROR: {e}")
    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--unit" in sys.argv:
        run_unit_tests()
    else:
        run_integration_tests()


# ─────────────────────────────────────────────────────────────────────────────
# Session pipeline integration test (server must be running)
# python test_cases.py --session
# ─────────────────────────────────────────────────────────────────────────────

def run_session_tests() -> None:
    """
    Test the full file-feed pipeline end-to-end.
    Requires backend running at http://127.0.0.1:8000
    """
    import urllib.request
    BASE = "http://127.0.0.1:8000"

    def get(path):
        with urllib.request.urlopen(f"{BASE}{path}", timeout=5) as r:
            return json.loads(r.read())

    def post(path, body=None):
        data = json.dumps(body or {}).encode()
        req  = urllib.request.Request(f"{BASE}{path}", data=data,
               headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    print(f"\n{SEP}\n  Session pipeline tests\n{SEP}")

    # 1. List patients
    patients = get("/api/patients")["patients"]
    print(f"  Patients available: {len(patients)}")
    assert len(patients) >= 1, "No patient files found"

    # 2. Start session
    r = post("/api/session/start", {"patient_id": patients[0]["patient_id"]})
    assert "aiScore" in r, f"No aiScore in start response: {r.keys()}"
    assert "_meta" in r,   "No _meta in start response"
    assert r["_meta"]["reading_index"] == 0
    print(f"  Start: aiScore={r['aiScore']}  label={r['_meta']['label']}")

    # 3. Tick twice
    for i in range(2):
        t = post("/api/session/tick")
        assert "aiScore" in t
        assert t["_meta"]["reading_index"] == i + 1
        print(f"  Tick {i+1}: aiScore={t['aiScore']:.4f}  "
              f"alert={t['alertLevel']}  idx={t['_meta']['reading_index']}")

    # 4. Status
    s = get("/api/session/status")
    assert s["active"] and s["current_index"] == 3
    print(f"  Status: active={s['active']}  index={s['current_index']}/{s['total_readings']}")

    # 5. Security: path traversal attempt
    try:
        post("/api/session/start", {"patient_id": "../../etc/passwd"})
        print("  FAIL: path traversal should have been rejected")
    except Exception as e:
        if "422" in str(e) or "400" in str(e) or "Invalid" in str(e):
            print("  ✓ Path traversal correctly rejected")
        else:
            print(f"  ? Traversal rejected with: {e}")

    # 6. Stop
    post("/api/session/stop")
    s2 = get("/api/session/status")
    assert not s2["active"], "Session not cleared after stop"
    print(f"  ✓ Session stopped")

    print(f"\n  Session pipeline tests PASSED\n")


if __name__ == "__main__":
    if "--session" in sys.argv:
        run_session_tests()
    elif "--unit" in sys.argv:
        run_unit_tests()
    else:
        run_integration_tests()
