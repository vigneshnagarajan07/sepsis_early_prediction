"""
Patient JSON stream manager for the Sepsis AI backend.

This module deliberately stays outside the model pipeline. It only loads
backend-hosted patient JSON files and converts each reading into the same
payload shape accepted by /api/predict.
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
import time
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATIENTS_DIR = os.path.join(SCRIPT_DIR, "data", "patients")

HISTORY_BUFFER = 20
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024

VITAL_KEYS = [
    "hr",
    "map",
    "resp",
    "temp",
    "o2sat",
    "urineOutput",
    "gcs",
    "systolicBp",
]

LAB_KEYS = [
    "lactate",
    "pct",
    "wbc",
    "platelets",
    "creatinine",
    "bilirubin",
    "crp",
    "dengueNS1",
    "malariaRDT",
]

DEFAULT_LABS: dict[str, dict[str, Any]] = {
    key: {"value": 0.0, "performed": False} for key in LAB_KEYS
}

_lock = threading.Lock()
_sessions: dict[str, dict[str, Any]] = {}


def list_patients() -> list[dict[str, Any]]:
    """Return metadata for valid patient JSON files."""
    os.makedirs(PATIENTS_DIR, exist_ok=True)
    patients: list[dict[str, Any]] = []

    for path in sorted(glob.glob(os.path.join(PATIENTS_DIR, "*.json"))):
        name = os.path.basename(path)
        if name.startswith(".") or name.startswith("~") or not os.path.isfile(path):
            continue
        try:
            patient = _load_patient_file(os.path.splitext(name)[0], validate_readings=False)
            readings = patient["readings"]
            patients.append({
                "patient_id": patient["patient_id"],
                "label": patient["label"],
                "description": patient.get("description", ""),
                "total_readings": int(patient.get("total_readings", len(readings))),
                "interval_minutes": float(patient.get("interval_minutes", 1)),
                "has_labs": any(_reading_has_labs(r) for r in readings),
            })
        except Exception:
            continue

    return patients


def start_session(patient_id: str, session_id: str) -> dict[str, Any]:
    """Load a patient JSON file and return the first prediction payload."""
    patient = _load_patient_file(patient_id, validate_readings=True)
    readings = patient["readings"]
    if not readings:
        raise ValueError(f"Patient {patient_id} has no readings.")

    interval_minutes = float(patient.get("interval_minutes", 1))
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than 0.")

    with _lock:
        _sessions[session_id] = {
            "patient_id": patient["patient_id"],
            "label": patient["label"],
            "description": patient.get("description", ""),
            "demographics": patient["demographics"],
            "readings": readings,
            "total": len(readings),
            "interval_minutes": interval_minutes,
            "next_index": 0,
            "history": {key: [] for key in VITAL_KEYS},
            "lab_history": {key: [] for key in LAB_KEYS},
            "last_tick": time.time(),
        }
        return _next_payload_locked(session_id)


def tick(session_id: str) -> dict[str, Any]:
    """Advance the active session by one reading."""
    with _lock:
        if session_id not in _sessions:
            raise RuntimeError(f"No active session for {session_id}. Call /api/session/start first.")
        _sessions[session_id]["last_tick"] = time.time()
        return _next_payload_locked(session_id)


def session_status(session_id: str) -> dict[str, Any]:
    """Return active session metadata without advancing."""
    with _lock:
        sess = _sessions.get(session_id)
        if sess is None:
            return {"active": False}
        return {
            "active": True,
            "patient_id": sess["patient_id"],
            "label": sess["label"],
            "current_index": sess["next_index"],
            "total_readings": sess["total"],
            "interval_minutes": sess["interval_minutes"],
            "done": sess["next_index"] >= sess["total"],
        }


def stop_session(session_id: str) -> None:
    """Clear the active session."""
    with _lock:
        if session_id in _sessions:
            del _sessions[session_id]


def cleanup_sessions(max_age_seconds: int = 3600) -> int:
    """Remove sessions that haven't been ticked for max_age_seconds."""
    now = time.time()
    with _lock:
        to_delete = [
            sid for sid, sess in _sessions.items()
            if now - sess.get("last_tick", 0) > max_age_seconds
        ]
        for sid in to_delete:
            del _sessions[sid]
        return len(to_delete)


def _next_payload_locked(session_id: str) -> dict[str, Any]:
    sess = _sessions[session_id]
    idx = int(sess["next_index"])
    total = int(sess["total"])
    if idx >= total:
        raise StopIteration("Session complete. All readings have been served.")

    reading = sess["readings"][idx]
    history = {key: list(values) for key, values in sess["history"].items()}
    previous_labs = _previous_labs_locked(session_id)
    vitals = _normalize_vitals(reading["vitals"])
    labs = _normalize_labs(reading.get("labs", {}))

    for key in VITAL_KEYS:
        sess["history"][key] = (sess["history"][key] + [vitals[key]])[-HISTORY_BUFFER:]

    for key in LAB_KEYS:
        lab = labs[key]
        if lab.get("performed", False):
            sess["lab_history"][key].append({
                "value": float(lab.get("value", 0.0)),
                "performed": True,
            })
        else:
            sess["lab_history"][key].append({"value": 0.0, "performed": False})
        sess["lab_history"][key] = sess["lab_history"][key][-HISTORY_BUFFER:]

    sess["next_index"] = idx + 1

    interval_minutes = float(sess["interval_minutes"])
    return {
        "vitals": vitals,
        "labs": labs,
        "demographics": sess["demographics"],
        "vitalsHistory": history,
        "previousLabs": previous_labs,
        "intervalSeconds": interval_minutes * 60.0,
        "_meta": {
            "patient_id": sess["patient_id"],
            "label": sess["label"],
            "description": sess["description"],
            "reading_index": idx,
            "total_readings": total,
            "timestamp_label": reading.get("timestamp_label", f"T+{idx * interval_minutes:g}min"),
            "interval_minutes": interval_minutes,
            "done": idx >= total - 1,
            "has_labs": _reading_has_labs(reading),
        },
    }


def _previous_labs_locked(session_id: str) -> dict[str, dict[str, Any]]:
    sess = _sessions[session_id]
    previous: dict[str, dict[str, Any]] = {}
    for key, entries in sess["lab_history"].items():
        for entry in reversed(entries):
            if entry.get("performed", False):
                previous[key] = {
                    "value": float(entry.get("value", 0.0)),
                    "performed": True,
                }
                break
    return previous


def _load_patient_file(patient_id: str, validate_readings: bool) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", patient_id):
        raise ValueError("Invalid patient_id. Use letters, digits, underscores, or hyphens only.")

    path = os.path.join(PATIENTS_DIR, f"{patient_id}.json")
    root = os.path.realpath(PATIENTS_DIR)
    resolved = os.path.realpath(path)
    if not resolved.startswith(root + os.sep):
        raise PermissionError("Patient path is outside the patient data directory.")
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Patient file not found: {patient_id}")
    if os.path.getsize(resolved) > MAX_FILE_SIZE_BYTES:
        raise ValueError("Patient file is too large.")

    with open(resolved, "r", encoding="utf-8") as f:
        patient = json.load(f)

    required = {"patient_id", "label", "readings", "demographics"}
    missing = required - set(patient)
    if missing:
        raise ValueError(f"Patient file missing required keys: {sorted(missing)}")
    if patient["patient_id"] != patient_id:
        raise ValueError("Patient file id does not match requested patient_id.")
    if not isinstance(patient["readings"], list):
        raise ValueError("Patient readings must be a list.")

    if validate_readings:
        for i, reading in enumerate(patient["readings"]):
            if not isinstance(reading, dict) or "vitals" not in reading:
                raise ValueError(f"Reading {i} must contain vitals.")
            _normalize_vitals(reading["vitals"])
            _normalize_labs(reading.get("labs", {}))

    return patient


def _normalize_vitals(raw: dict[str, Any]) -> dict[str, float]:
    missing = [key for key in VITAL_KEYS if key not in raw]
    if missing:
        raise ValueError(f"Vitals missing required keys: {missing}")

    vitals = {key: float(raw[key]) for key in VITAL_KEYS}
    if not 30 <= vitals["hr"] <= 250:
        raise ValueError("Heart rate out of supported range.")
    if not 30 <= vitals["map"] <= 160:
        raise ValueError("MAP out of supported range.")
    if not 4 <= vitals["resp"] <= 60:
        raise ValueError("Respiratory rate out of supported range.")
    if not 34 <= vitals["temp"] <= 42:
        raise ValueError("Temperature out of supported range.")
    if not 60 <= vitals["o2sat"] <= 100:
        raise ValueError("O2 saturation out of supported range.")
    if not 0 <= vitals["urineOutput"] <= 5:
        raise ValueError("Urine output out of supported range.")
    if not 3 <= vitals["gcs"] <= 15:
        raise ValueError("GCS out of supported range.")
    if not 50 <= vitals["systolicBp"] <= 220:
        raise ValueError("Systolic BP out of supported range.")
    if vitals["map"] > vitals["systolicBp"]:
        raise ValueError("MAP cannot exceed systolicBp.")

    if "motionArtifact" in raw:
        vitals["motionArtifact"] = float(raw["motionArtifact"])
    if "sensorDetached" in raw:
        vitals["sensorDetached"] = float(raw["sensorDetached"])
    return vitals


def _normalize_labs(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    labs = {key: dict(value) for key, value in DEFAULT_LABS.items()}
    for key, value in raw.items():
        if key not in labs or not isinstance(value, dict):
            continue
        labs[key] = {
            "value": float(value.get("value", 0.0)),
            "performed": bool(value.get("performed", False)),
        }
    return labs


def _reading_has_labs(reading: dict[str, Any]) -> bool:
    labs = reading.get("labs", {})
    return any(isinstance(v, dict) and v.get("performed", False) for v in labs.values())
