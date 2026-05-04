    """
    test_cases.py – Three clinical sanity-check payloads
    Case 1 → CRITICAL  (severe sepsis: high HR, low BP, high RR, high lactate, low GCS)
    Case 2 → WARNING   (dengue thrombocytopenia + low WBC; no labs + prior abx + referral)
    Case 3 → NONE      (healthy vitals, all normal labs, no risk factors)
    """

    import json
    import urllib.request

    URL = "http://127.0.0.1:8000/api/predict"

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
                "diabetes": True, "ckd": True,
                "priorAntibiotics": False, "referredFromOutside": False,
                "gramNegativeRisk": True, "malariaEndemic": False, "dengueEndemic": False,
                "covidPrevalence": False
            }
        },

        "Case 2 — WARNING (dengue thrombocytopenia, informative missingness)": {
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
                "priorAntibiotics": True, "referredFromOutside": True,
                "gramNegativeRisk": False, "malariaEndemic": False, "dengueEndemic": True,
                "covidPrevalence": False
            }
        },

        "Case 3 — NONE (healthy, no risk)": {
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
                "priorAntibiotics": False, "referredFromOutside": False,
                "gramNegativeRisk": False, "malariaEndemic": False, "dengueEndemic": False,
                "covidPrevalence": False
            }
        }
    }

    SEP = "─" * 60

    def run_case(label, payload):
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    for label, payload in CASES.items():
        print(f"\n{SEP}")
        print(f"  {label}")
        print(SEP)
        try:
            r = run_case(label, payload)
            print(f"  aiScore      : {r['aiScore']}")
            print(f"  qsofaScore   : {r['qsofaScore']}")
            print(f"  alertLevel   : {r['alertLevel'].upper()}")
            print(f"  modelScores  : {r['modelScores']}")
            print(f"  labScenario  : {r['featureSummary']['labScenario']}")
            print(f"  shockIndex   : {r['featureSummary']['shockIndex']}")
            print(f"  crt          : {r['featureSummary']['crt']}")
            print(f"  Top SHAP drivers:")
            for d in r.get("shapDrivers", [])[:5]:
                bar = "▲" if d["shap"] > 0 else "▼"
                print(f"    {bar}  {d['feature']:<28} val={d['value']}  shap={d['shap']}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{SEP}\n")
