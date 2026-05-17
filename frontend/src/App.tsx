/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { 
  Activity, 
  ChevronRight,
  Eye,
  EyeOff,
  Radio,
  Clock,
  Beaker,
  TrendingUp,
  TrendingDown,
  History,
  Database,
  PlayCircle,
  StopCircle,
  ChevronDown
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import * as d3 from 'd3';

// --- Types ---

interface PatientDemographics {
  age: number;
  gender: 'Male' | 'Female' | 'Other';
  bmi: number;
  diabetes: boolean;
  ckd: boolean; // Chronic Kidney Disease
  cirrhosis: boolean;
  malignancy: boolean;
  immunosuppression: boolean;
  priorAntibiotics: boolean; // Critical Risk Factor
  referredFromOutside: boolean; // Critical Risk Factor
  gramNegativeRisk: boolean; // Maps to AMR_Resistance in backend
  malariaEndemic: boolean;
  dengueEndemic: boolean;
}

interface Vitals {
  hr: number;
  map: number;
  resp: number;
  temp: number;
  o2sat: number;
  urineOutput: number;
  gcs: number;
  systolicBp: number;
}

type VitalsHistory = {
  [K in keyof Vitals]: number[];
};

interface LabTest {
  value: number;
  performed: boolean;
}

interface Labs {
  lactate: LabTest;
  pct: LabTest; // Procalcitonin
  wbc: LabTest;
  platelets: LabTest;
  creatinine: LabTest;
  bilirubin: LabTest;
  crp: LabTest; // FIX #21: CRP was in backend Labs model but missing from frontend interface
  dengueNS1: LabTest;
  malariaRDT: LabTest;
}

// Patient file metadata from /api/patients
interface PatientMeta {
  patient_id: string;
  label: string;
  description: string;
  total_readings: number;
  interval_minutes: number;
  has_labs: boolean;
}

// --- Components ---

export default function SepsisDashboard() {
  const [sessionId] = useState(() => crypto.randomUUID());

  // 1. Patient Demographics (Sidebar)
  const [demographics, setDemographics] = useState<PatientDemographics>({
    age: 45,
    gender: 'Male',
    bmi: 24.5,
    diabetes: false,
    ckd: false,
    cirrhosis: false,
    malignancy: false,
    immunosuppression: false,
    priorAntibiotics: false,
    referredFromOutside: false,
    gramNegativeRisk: false,
    malariaEndemic: false,
    dengueEndemic: false,
  });

  // 2. Vitals (Main Section 1)
  const [vitals, setVitals] = useState<Vitals>({
    hr: 85,
    map: 90,
    resp: 18,
    temp: 37.0,
    o2sat: 98,
    urineOutput: 0.8,
    gcs: 15,
    systolicBp: 120,
  });

  const [vitalsHistory, setVitalsHistory] = useState<VitalsHistory>({
    hr: [80, 82, 85, 84, 85],
    map: [88, 89, 90, 91, 90],
    resp: [16, 17, 18, 17, 18],
    temp: [36.8, 36.9, 37.0, 37.0, 37.0],
    o2sat: [97, 98, 98, 99, 98],
    urineOutput: [0.7, 0.8, 0.8, 0.8, 0.8],
    gcs: [15, 15, 15, 15, 15],
    systolicBp: [118, 120, 120, 122, 120],
  });

  // 3. Labs (Main Section 2)
  // FIX: all performed:false — blank form = no labs drawn.
  // Values pre-filled as sensible defaults for when nurse toggles performed:true.
  // Sending performed:false to backend triggers median imputation, not value=0.
  const [labs, setLabs] = useState<Labs>({
    lactate: { value: 1.2, performed: false },
    pct: { value: 0.05, performed: false },
    wbc: { value: 8.5, performed: false },
    platelets: { value: 250, performed: false },
    creatinine: { value: 1.0, performed: false },
    bilirubin: { value: 0.5, performed: false },
    crp: { value: 5.0, performed: false },
    dengueNS1: { value: 0, performed: false },
    malariaRDT: { value: 0, performed: false },
  });

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  // ── File-feed pipeline state ──
  const [isFileFed, setIsFileFed]     = useState(false);
  const [patients, setPatients]       = useState<PatientMeta[]>([]);
  const [activePatient, setActivePatient] = useState<PatientMeta | null>(null);
  const [feedProgress, setFeedProgress]   = useState({ current: 0, total: 0 });
  const [showPatientPicker, setShowPatientPicker] = useState(false);
  const feedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // BUG FIX: track in-flight requests to prevent concurrent stacking during simulation
  const analysisInFlight = useRef(false);
  // BUG FIX: surface API errors to the clinician instead of only console.error
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'live' | 'historical'>('live');
  const [timeRange, setTimeRange] = useState<'6h' | '12h' | '24h'>('6h');
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [result, setResult] = useState<{
    aiScore: number;
    qsofaScore: number;
    alertLevel: 'none' | 'warning' | 'critical';
    attentionWeights: number[];
    modelScores: Record<string, number | null>; // FIX #22: values can be null (e.g. xgb_lab when no labs)
    featureSummary: { crt: number; crtSynthetic?: boolean; shockIndex: number; labScenario: string; hasLabs?: boolean; calibrated?: boolean; xgbEligible?: boolean };
    // DEMO: data is 225× compressed (15-min clinical readings → 4s demo playback)
    shapDrivers: Array<{ feature: string; value: number; shap: number }>;
    confidenceScore: number;
    dataQualityWarnings: string[];
    deltaInfo?: {
      sourceReadings: number; intervalSec: number;
      deltaHR: number; deltaMAP: number; deltaResp: number; deltaTemp: number;
      deltaLactate: number; deltaCreatinine: number;
    };
  } | null>(null);

  // Previous lab draw — sent to backend to compute Delta_3h_Lactate etc.
  const [previousLabs, setPreviousLabs] = useState<Labs | null>(null);

  // --- Logic ---

  // qSOFA is computed server-side with full clinical logic; no local duplicate needed.

  const runAnalysis = useCallback(async () => {
    // BUG FIX: skip if a request is already in flight (prevents concurrent stacking in sim mode)
    if (analysisInFlight.current) return;
    analysisInFlight.current = true;
    setIsAnalyzing(true);
    setResult(null);
    setAnalysisError(null);

    const labsSnapshot = labs;

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-Session-ID': sessionId
        },
        body: JSON.stringify({
          vitals,
          labs,
          demographics,
          vitalsHistory,
          previousLabs: previousLabs ?? undefined,
          intervalSeconds: isSimulating ? 5 : 0,
        })
      });

      if (!response.ok) {
        let errorMsg = `Prediction failed (HTTP ${response.status})`;
        try {
          const errorJson = await response.json();
          // FastAPI error detail is in errorJson.detail
          const detail = errorJson?.detail ?? JSON.stringify(errorJson);
          errorMsg += `: ${detail}`;
        } catch {
          const errorText = await response.text();
          errorMsg += `: ${errorText.substring(0, 200)}`;
        }
        throw new Error(errorMsg);
      }

      const analysis = await response.json();
      setResult(analysis);
      setPreviousLabs(labsSnapshot);
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      console.error("Inference Error:", msg);
      // BUG FIX: show error to clinician in the UI, not just the console
      setAnalysisError(msg);
    } finally {
      setIsAnalyzing(false);
      analysisInFlight.current = false;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vitals, labs, demographics, vitalsHistory, previousLabs, isSimulating]);

  // ─── File-Feed Pipeline ──────────────────────────────────────────
  // Loads patient list on mount
  useEffect(() => {
    fetch('/api/patients', { headers: { 'X-Session-ID': sessionId } })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(d => setPatients(d.patients || []))
      .catch(err => {
        console.error('Failed to load patient list:', err);
        setAnalysisError('Backend unavailable — patient files could not be loaded.');
      });
  }, [sessionId]);

  // Apply a tick response — updates ALL state from backend file reading
  const applyTickResponse = useCallback((data: any) => {
    if (!data || data.done) return;

    if (data.demographics) {
      setDemographics(data.demographics);
    }

    if (data.vitalsHistory) {
      setVitalsHistory(data.vitalsHistory);
    }

    // FIX F2: optional chaining + shape guard
    if (data?.vitals && typeof data.vitals.hr === "number") {
      setVitals({
        hr:          data.vitals.hr,
        map:         data.vitals.map,
        resp:        data.vitals.resp,
        temp:        data.vitals.temp,
        o2sat:       data.vitals.o2sat,
        systolicBp:  data.vitals.systolicBp,
        gcs:         data.vitals.gcs,
        urineOutput: data.vitals.urineOutput,
      });
    }

    if (data?.labs && typeof data.labs === "object") {
      setLabs(prev => ({
        ...prev,
        ...Object.fromEntries(
          Object.entries(data.labs).map(([k, v]) => [k, v as any])
        ),
      }));
    }

    if (data?.aiScore !== undefined) setResult(data);

    if (data?._meta) {
      setFeedProgress({
        current: data._meta.reading_index + 1,
        total:   data._meta.total_readings,
      });
    }
    setLastUpdate(new Date());
  }, []);

  // Start a file-feed session with a chosen patient
  const startFileFeed = useCallback(async (patientId: string) => {
    // Stop any running simulation
    setIsSimulating(false);
    if (feedTimerRef.current) {
      clearTimeout(feedTimerRef.current);
      feedTimerRef.current = null;
    }

    setIsAnalyzing(true);
    setAnalysisError(null);
    try {
      const res = await fetch('/api/session/start', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-Session-ID': sessionId
        },
        body: JSON.stringify({ patient_id: patientId }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Failed to start session');
      }
      const data = await res.json();
      applyTickResponse(data);

      const patient = patients.find(p => p.patient_id === patientId) || null;
      setActivePatient(patient);
      setIsFileFed(true);
      setShowPatientPicker(false);
      // NOTE: We no longer start the tick loop here.
      // The user must click "Start File Feed" to begin playback.

    } catch (e: any) {
      setAnalysisError(e.message || 'Failed to start file feed');
    } finally {
      setIsAnalyzing(false);
    }
  }, [patients, applyTickResponse, sessionId]);

  // Handle File Feed Polling
  useEffect(() => {
    if (isFileFed && isSimulating) {
      const pollMs = 2000;
      const tick = async () => {
        try {
          const tickRes = await fetch('/api/session/tick', { 
            method: 'POST',
            headers: { 'X-Session-ID': sessionId }
          });
          const tickData = await tickRes.json();
          if (tickData.done) {
            setIsSimulating(false);
            return;
          }
          applyTickResponse(tickData);
          feedTimerRef.current = setTimeout(tick, pollMs);
        } catch {
          // network blip — keep polling
          feedTimerRef.current = setTimeout(tick, pollMs);
        }
      };
      feedTimerRef.current = setTimeout(tick, pollMs);
    }
    return () => {
      if (feedTimerRef.current) {
        clearTimeout(feedTimerRef.current);
        feedTimerRef.current = null;
      }
    };
  }, [isFileFed, isSimulating, applyTickResponse, sessionId]);

  // Stop file-feed session
  const stopFileFeed = useCallback(() => {
    setIsSimulating(false);
    if (feedTimerRef.current) {
      clearTimeout(feedTimerRef.current);
      feedTimerRef.current = null;
    }
    setIsFileFed(false);
    setActivePatient(null);
    setFeedProgress({ current: 0, total: 0 });
    
    // Reset to defaults
    setDemographics({
      age: 45, gender: 'Male', bmi: 24.5, diabetes: false, ckd: false,
      cirrhosis: false, malignancy: false, immunosuppression: false,
      priorAntibiotics: false, referredFromOutside: false,
      gramNegativeRisk: false, malariaEndemic: false, dengueEndemic: false
    });
    setVitals({ hr: 85, map: 90, resp: 18, temp: 37.0, o2sat: 98, urineOutput: 0.8, gcs: 15, systolicBp: 120 });
    setVitalsHistory({
      hr: [80, 82, 85, 84, 85], map: [88, 89, 90, 91, 90], resp: [16, 17, 18, 17, 18],
      temp: [36.8, 36.9, 37.0, 37.0, 37.0], o2sat: [97, 98, 98, 99, 98],
      urineOutput: [0.7, 0.8, 0.8, 0.8, 0.8], gcs: [15, 15, 15, 15, 15], systolicBp: [118, 120, 120, 122, 120]
    });
    setResult(null);

    fetch('/api/session/stop', { 
      method: 'POST',
      headers: { 'X-Session-ID': sessionId }
    }).catch(() => {});
  }, [sessionId]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (feedTimerRef.current) clearTimeout(feedTimerRef.current);
    };
  }, []);
  // ─────────────────────────────────────────────────────────────────

  // Helper for lab inputs
  const updateLab = (key: keyof Labs, field: 'value' | 'performed', val: any) => {
    setLabs(prev => ({
      ...prev,
      [key]: { ...prev[key], [field]: val }
    }));
  };

  const updateVital = (key: keyof Vitals, val: number) => {
    setVitals(prev => ({ ...prev, [key]: val }));
    setVitalsHistory(prev => ({
      ...prev,
      [key]: [...prev[key].slice(-19), val]
    }));
  };

  // --- Real-time Simulation Engine ---
  // FIX #23: store latest runAnalysis in a ref so the interval closure always
  // calls the current version without listing runAnalysis as a dep (which would
  // restart the interval on every vitals state change during simulation).
  const runAnalysisRef = useRef(runAnalysis);
  useEffect(() => { runAnalysisRef.current = runAnalysis; }, [runAnalysis]);

  // FIX #23: use a ref to hold the latest runAnalysis so the interval
  // is NOT recreated every time runAnalysis changes (dep churn).
  // The interval is created once when simulation starts and reads the
  // current runAnalysis via ref on every tick.

  // 1. Data Drift (Vitals Only) + periodic analysis
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isSimulating && !isFileFed) {
      interval = setInterval(() => {
        setVitals(prev => {
          const nextVitals = {
            ...prev,
            hr: Math.min(180, Math.max(40, prev.hr + (Math.random() * 6 - 3))),
            resp: Math.min(45, Math.max(8, prev.resp + (Math.random() * 2 - 1))),
            temp: parseFloat(Math.min(41, Math.max(35, prev.temp + (Math.random() * 0.2 - 0.1))).toFixed(1)),
            systolicBp: Math.min(190, Math.max(60, prev.systolicBp + (Math.random() * 6 - 3))),
            map: Math.min(140, Math.max(40, prev.map + (Math.random() * 4 - 2)))
          };

          setVitalsHistory(hPrev => {
            const nextHistory = { ...hPrev };
            (Object.keys(nextVitals) as Array<keyof Vitals>).forEach(key => {
              nextHistory[key] = [...hPrev[key].slice(-19), nextVitals[key]];
            });
            return nextHistory;
          });

          return nextVitals;
        });

        setLastUpdate(new Date());
        // Use ref to call current runAnalysis without dep-churn (FIX #23)
        runAnalysisRef.current();
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isSimulating, isFileFed]); // FIX #23: runAnalysis removed from deps — accessed via ref

  // 2. Auto-Analysis sync — trigger once per simulation tick, not on every state change
  // BUG FIX: original deps array [vitals, labs, demographics, isSimulating] caused
  // runAnalysis to fire on every keystroke in the UI when simulating.
  // Now we only trigger when the simulation flag itself changes to true.
  useEffect(() => {
    if (isSimulating && !isFileFed) {
      runAnalysis();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isSimulating, isFileFed]);

  return (
    <div className="flex flex-col md:flex-row min-h-screen bg-brand-bg text-brand-text font-sans selection:bg-brand-primary/10">
      {/* Sidebar - Patient History */}
      <aside className="w-full md:w-[260px] bg-white border-r border-brand-border p-5 flex flex-col gap-6 shadow-sm overflow-y-auto">
        <header className="flex items-center gap-2">
          <div className="bg-brand-primary p-2 rounded text-white shrink-0">
            <Activity size={20} />
          </div>
          <div>
            <h1 className="font-extrabold text-base tracking-tight text-brand-primary leading-tight">Early Prediction of Sepsis</h1>
            <p className="text-[9px] uppercase font-bold text-brand-muted tracking-widest leading-none">Diagnostic Intelligence</p>
          </div>
        </header>

        <section className="space-y-4">
          <div className="pb-1 border-b border-brand-border">
            <h2 className="text-[11px] font-bold uppercase tracking-wider text-brand-muted">Patient Demographics</h2>
          </div>

          <div className="space-y-4">
            <div className="input-group flex flex-col gap-1">
              <label className="text-[11px] font-bold text-brand-text">Patient ID</label>
              <input type="text" value={activePatient ? activePatient.patient_id : "P-99201"} readOnly className="w-full bg-[#fafafa] border border-brand-border rounded px-2 py-1.5 text-[13px] outline-none" />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="input-group flex flex-col gap-1">
                <label className="text-[11px] font-bold text-brand-text">Age</label>
                <input 
                  type="number" 
                  value={demographics.age}
                  onChange={e => setDemographics({...demographics, age: parseInt(e.target.value) || 0})}
                  className="w-full bg-[#fafafa] border border-brand-border rounded px-2 py-1.5 text-[13px] outline-none"
                />
              </div>
              <div className="input-group flex flex-col gap-1">
                <label className="text-[11px] font-bold text-brand-text">BMI</label>
                <input 
                  type="number" 
                  step="0.1"
                  value={demographics.bmi}
                  onChange={e => setDemographics({...demographics, bmi: parseFloat(e.target.value) || 0})}
                  className="w-full bg-[#fafafa] border border-brand-border rounded px-2 py-1.5 text-[13px] outline-none"
                />
              </div>
            </div>

            <div className="input-group flex flex-col gap-1">
              <label className="text-[11px] font-bold text-brand-text">Gender</label>
              <select 
                value={demographics.gender}
                onChange={e => setDemographics({...demographics, gender: e.target.value as any})}
                className="w-full bg-[#fafafa] border border-brand-border rounded px-2 py-1.5 text-[13px] outline-none appearance-none cursor-pointer"
              >
                <option>Male</option>
                <option>Female</option>
                <option>Other</option>
              </select>
            </div>

            <div className="pt-2">
              <div className="text-[11px] font-bold uppercase tracking-wider text-brand-muted mb-3 pb-1 border-b border-brand-border">Risk Context</div>
              
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] font-bold text-brand-text">Diabetes</label>
                    <select 
                      value={demographics.diabetes ? 'Yes' : 'No'}
                      onChange={e => setDemographics({...demographics, diabetes: e.target.value === 'Yes'})}
                      className="w-full bg-[#fafafa] border border-brand-border rounded px-2 py-1.5 text-[13px] outline-none appearance-none cursor-pointer"
                    >
                      <option>No</option>
                      <option>Yes</option>
                    </select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] font-bold text-brand-text">CKD</label>
                    <select 
                      value={demographics.ckd ? 'Yes' : 'No'}
                      onChange={e => setDemographics({...demographics, ckd: e.target.value === 'Yes'})}
                      className="w-full bg-[#fafafa] border border-brand-border rounded px-2 py-1.5 text-[13px] outline-none appearance-none cursor-pointer"
                    >
                      <option>No</option>
                      <option>Yes</option>
                    </select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] font-bold text-brand-text">Cirrhosis</label>
                    <select 
                      value={demographics.cirrhosis ? 'Yes' : 'No'}
                      onChange={e => setDemographics({...demographics, cirrhosis: e.target.value === 'Yes'})}
                      className="w-full bg-[#fafafa] border border-brand-border rounded px-2 py-1.5 text-[13px] outline-none appearance-none cursor-pointer"
                    >
                      <option>No</option>
                      <option>Yes</option>
                    </select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] font-bold text-brand-text">Malignancy</label>
                    <select 
                      value={demographics.malignancy ? 'Yes' : 'No'}
                      onChange={e => setDemographics({...demographics, malignancy: e.target.value === 'Yes'})}
                      className="w-full bg-[#fafafa] border border-brand-border rounded px-2 py-1.5 text-[13px] outline-none appearance-none cursor-pointer"
                    >
                      <option>No</option>
                      <option>Yes</option>
                    </select>
                  </div>
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-[11px] font-bold text-brand-text">Immunosuppression</label>
                  <select 
                    value={demographics.immunosuppression ? 'Yes' : 'No'}
                    onChange={e => setDemographics({...demographics, immunosuppression: e.target.value === 'Yes'})}
                    className="w-full bg-[#fafafa] border border-brand-border rounded px-2 py-1.5 text-[13px] outline-none appearance-none cursor-pointer"
                  >
                    <option>No</option>
                    <option>Yes</option>
                  </select>
                </div>

                {/* Critical Risk Factors */}
                <div className="flex flex-col gap-1">
                  <label className="text-[11px] font-bold text-brand-text">Prior Antibiotics (Unprescribed)</label>
                  <select 
                    value={demographics.priorAntibiotics ? 'Yes' : 'No'}
                    onChange={e => setDemographics({...demographics, priorAntibiotics: e.target.value === 'Yes'})}
                    className={`w-full bg-[#fafafa] border rounded px-2 py-1.5 text-[13px] outline-none appearance-none cursor-pointer ${demographics.priorAntibiotics ? 'border-brand-danger shadow-[0_0_8px_rgba(229,62,62,0.1)]' : 'border-brand-border'}`}
                  >
                    <option>No</option>
                    <option>Yes</option>
                  </select>
                  {demographics.priorAntibiotics && <span className="inline-block self-start px-1.5 py-0.5 mt-1 bg-red-50 text-brand-danger border border-red-100 rounded text-[9px] font-bold uppercase">Severe Risk Escalation</span>}
                </div>

                <div className="flex flex-col gap-1">
                  <label className="text-[11px] font-bold text-brand-text">Outside Referral</label>
                  <select 
                    value={demographics.referredFromOutside ? 'Yes' : 'No'}
                    onChange={e => setDemographics({...demographics, referredFromOutside: e.target.value === 'Yes'})}
                    className={`w-full bg-[#fafafa] border rounded px-2 py-1.5 text-[13px] outline-none appearance-none cursor-pointer ${demographics.referredFromOutside ? 'border-brand-danger shadow-[0_0_8px_rgba(229,62,62,0.1)]' : 'border-brand-border'}`}
                  >
                    <option>No</option>
                    <option>Yes</option>
                  </select>
                  {demographics.referredFromOutside && <span className="inline-block self-start px-1.5 py-0.5 mt-1 bg-red-50 text-brand-danger border border-red-100 rounded text-[9px] font-bold uppercase">Severe Risk Escalation</span>}
                </div>

                <div className="pt-2">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-brand-muted mb-2 pb-1 border-b border-brand-border">Epidemiological Context</div>
                  <div className="grid grid-cols-1 gap-2">
                    <RiskToggle 
                      label="Gram -ve Risk" 
                      active={demographics.gramNegativeRisk} 
                      onClick={() => setDemographics({...demographics, gramNegativeRisk: !demographics.gramNegativeRisk})} 
                    />
                    <RiskToggle 
                      label="Malaria Endemic" 
                      active={demographics.malariaEndemic} 
                      onClick={() => setDemographics({...demographics, malariaEndemic: !demographics.malariaEndemic})} 
                    />
                    <RiskToggle 
                      label="Dengue Endemic" 
                      active={demographics.dengueEndemic} 
                      onClick={() => setDemographics({...demographics, dengueEndemic: !demographics.dengueEndemic})} 
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <div className="mt-auto pt-4 border-t border-brand-border">
           <div className="text-[10px] text-brand-muted font-medium text-center">
              Hospital ID: IND-BLR-001
           </div>
        </div>
      </aside>

      {/* Main Area */}
      <main className="flex-1 p-6 overflow-y-auto space-y-5 max-w-6xl mx-auto w-full flex flex-col">
        <header className="flex justify-between items-center shrink-0">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-extrabold text-brand-primary tracking-tight">Clinical Dashboard</h1>
            <div className="flex items-center gap-2 px-3 py-1 bg-[#edf2f7] rounded-full text-[12px] font-bold text-brand-text">
              {isSimulating && <div className="w-2 h-2 rounded-full bg-brand-danger animate-pulse" />}
              <span className={isSimulating ? 'text-brand-danger' : ''}>
                {isSimulating ? 'Streaming Real-time Vitals' : 'Diagnostic Mode Active'}
              </span>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex bg-[#edf2f7] p-1 rounded-lg">
              <button 
                onClick={() => setViewMode('live')}
                className={`px-3 py-1.5 rounded-md text-[11px] font-bold transition-all ${viewMode === 'live' ? 'bg-white text-brand-primary shadow-sm' : 'text-brand-muted hover:text-brand-primary'}`}
              >
                Live View
              </button>
              <button 
                onClick={() => setViewMode('historical')}
                className={`px-3 py-1.5 rounded-md text-[11px] font-bold transition-all ${viewMode === 'historical' ? 'bg-white text-brand-primary shadow-sm' : 'text-brand-muted hover:text-brand-primary'}`}
              >
                Historical
              </button>
            </div>
            
            <button 
              onClick={() => setIsSimulating(!isSimulating)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded text-[11px] font-bold uppercase tracking-wider transition-all border ${
                isSimulating 
                  ? 'bg-brand-danger text-white border-brand-danger hover:bg-brand-danger/90' 
                  : 'bg-white text-brand-primary border-brand-primary hover:bg-brand-primary/10'
              }`}
            >
              {isSimulating ? <StopCircle size={14} /> : <PlayCircle size={14} />}
              {isFileFed 
                ? (isSimulating ? 'Pause File Feed' : 'Start File Feed') 
                : (isSimulating ? 'Stop Live Feed' : 'Start Live Feed')}
            </button>

            {isFileFed && (
              <button
                onClick={stopFileFeed}
                className="px-3 py-1.5 rounded text-[11px] font-bold uppercase tracking-wider border border-gray-300 bg-white text-gray-600 hover:bg-gray-100 flex items-center gap-2 transition-all"
              >
                <StopCircle size={14} className="text-gray-400" />
                Unload File
              </button>
            )}

            {!isFileFed && (
              <button
                onClick={() => { setIsSimulating(false); setShowPatientPicker(true); }}
                className="px-3 py-1.5 text-[12px] font-medium border border-blue-300 bg-blue-50 rounded text-blue-700 hover:bg-blue-100 transition-colors flex items-center gap-1.5"
              >
                <Database size={14} />
                Load Patient File
              </button>
            )}
            
            <div className="text-[12px] text-brand-muted flex items-center gap-1">
              <Clock size={12} />
              Synced: {lastUpdate.toLocaleTimeString()}
            </div>
          </div>
        </header>

        <div className="flex flex-col gap-6 flex-grow pb-10">
          {viewMode === 'live' ? (
            <>
              {/* Section 1: Live Vital Signs */}
              <section className="bg-white rounded-lg border border-brand-border shadow-[0_1px_3px_rgba(0,0,0,0.05)] relative z-10">
                <header className="px-4 py-3 bg-[#fcfcfc] border-b border-brand-border flex items-center justify-between rounded-t-lg">
                  <div className="flex items-center gap-2">
                    <Activity size={16} className="text-brand-primary" />
                    <h3 className="text-[13px] font-bold uppercase tracking-tight">Clinical Monitoring & Vital Signs</h3>
                    {isSimulating && (
                      <div className="flex items-center gap-1 px-1.5 py-0.5 bg-brand-danger/10 text-brand-danger text-[9px] font-bold uppercase rounded border border-brand-danger/20">
                        <Radio size={10} className="animate-pulse" />
                        Live Feed
                      </div>
                    )}
                  </div>
                  <span className="text-[10px] text-brand-muted font-bold uppercase tracking-widest">Real-time Telemetry</span>
                </header>

                <div className="p-4 bg-white">
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
                    <VitalInput label="Heart Rate (BPM)" value={vitals.hr} history={vitalsHistory.hr} min={40} max={200} onChange={v => updateVital('hr', v)} />
                    <VitalInput label="Resp Rate" value={vitals.resp} history={vitalsHistory.resp} min={8} max={50} onChange={v => updateVital('resp', v)} />
                    <VitalInput label="Temp (°C)" value={vitals.temp} history={vitalsHistory.temp} min={34} max={42} step={0.1} onChange={v => updateVital('temp', v)} />
                    <VitalInput label="O2 Sat (%)" value={vitals.o2sat} history={vitalsHistory.o2sat} min={70} max={100} onChange={v => updateVital('o2sat', v)} />
                    <VitalInput label="Systolic BP" value={vitals.systolicBp} history={vitalsHistory.systolicBp} min={60} max={200} onChange={v => updateVital('systolicBp', v)} />
                    <VitalInput label="MAP (mmHg)" value={vitals.map} history={vitalsHistory.map} min={40} max={140} onChange={v => updateVital('map', v)} />
                    <VitalInput label="Urine (mL/kg/h)" value={vitals.urineOutput} history={vitalsHistory.urineOutput} min={0} max={3.0} step={0.1} onChange={v => updateVital('urineOutput', v)} />
                    <VitalInput label="GCS Score" value={vitals.gcs} history={vitalsHistory.gcs} min={3} max={15} onChange={v => updateVital('gcs', v)} />
                    <div className="vital-input flex flex-col gap-1">
                      <span className="text-[10px] text-brand-muted uppercase font-bold">
                        Delta HR (3h)
                        {result?.deltaInfo && result.deltaInfo.sourceReadings > 1 && (
                          <span className="ml-1 text-brand-success opacity-70">●</span>
                        )}
                      </span>
                      <div className={`w-full border rounded p-2 text-[13px] font-mono flex items-center gap-1 ${
                        result?.deltaInfo && result.deltaInfo.sourceReadings > 1
                          ? 'bg-white border-brand-border'
                          : 'bg-[#edf2f7] border-brand-border text-brand-muted'
                      }`}>
                        {result?.deltaInfo && result.deltaInfo.sourceReadings > 1 ? (
                          <>
                            <span className={result.deltaInfo.deltaHR > 5 ? 'text-brand-danger' : result.deltaInfo.deltaHR < -5 ? 'text-brand-success' : ''}>
                              {result.deltaInfo.deltaHR > 0 ? '+' : ''}{result.deltaInfo.deltaHR}
                            </span>
                            <span className="text-[9px] text-brand-muted ml-auto">
                              {result.deltaInfo.sourceReadings}pts
                            </span>
                          </>
                        ) : (
                          <span className="text-brand-muted text-[11px]">— awaiting history</span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="mt-5 pt-4 border-t border-brand-border flex flex-col gap-3">
                    {/* BUG FIX: surface API errors to the clinician */}
                    {analysisError && (
                      <div className="px-4 py-3 bg-red-50 border border-red-200 rounded text-[12px] text-brand-danger leading-snug">
                        <span className="font-bold">Analysis error: </span>{analysisError}
                      </div>
                    )}
                    <div className="flex justify-end">
                      <button 
                        disabled={isAnalyzing}
                        onClick={runAnalysis}
                        className="px-6 py-2.5 bg-brand-primary hover:bg-brand-primary/90 disabled:bg-brand-primary/50 text-white rounded font-bold text-[13px] shadow-sm transition-all active:scale-95 flex items-center gap-2"
                      >
                        {isAnalyzing ? (
                          <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                        ) : (
                          <ChevronRight size={16} />
                        )}
                        Evaluate Sepsis Probability
                      </button>
                    </div>
                  </div>
                </div>
              </section>

              {/* Section 2: Laboratory Panels */}
              <section className="bg-white rounded-lg border border-brand-border shadow-[0_1px_3px_rgba(0,0,0,0.05)] relative z-0">
                <header className="px-4 py-3 bg-[#fcfcfc] border-b border-brand-border flex items-center justify-between rounded-t-lg">
                  <div className="flex items-center gap-2">
                    <Beaker size={16} className="text-brand-primary" />
                    <h3 className="text-[13px] font-bold uppercase tracking-tight">Diagnostic Laboratory Panels</h3>
                  </div>
                  <span className="text-[10px] text-brand-muted font-bold uppercase tracking-widest">Informative Missingness Active</span>
                </header>

                <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-4">
                  <div className="space-y-4">
                    <LabRow
                      label="Serum Lactate"
                      unit="mmol/L" refRange="< 2.0" refMin={0.1} refMax={2.0}
                      value={labs.lactate.value} performed={labs.lactate.performed}
                      min={0.1} max={20} step={0.1}
                      onValueChange={v => updateLab('lactate', 'value', v)}
                      onToggle={p => updateLab('lactate', 'performed', p)}
                    />
                    <LabRow
                      label="Procalcitonin"
                      unit="ng/mL" refRange="< 0.25" refMin={0} refMax={0.25}
                      value={labs.pct.value} performed={labs.pct.performed}
                      min={0.001} max={1000} step={0.01}
                      onValueChange={v => updateLab('pct', 'value', v)}
                      onToggle={p => updateLab('pct', 'performed', p)}
                    />
                    <LabRow
                      label="WBC Count"
                      unit="×10³/µL" refRange="4.0–11.0" refMin={4.0} refMax={11.0}
                      value={labs.wbc.value} performed={labs.wbc.performed}
                      min={0.1} max={200} step={0.1}
                      onValueChange={v => updateLab('wbc', 'value', v)}
                      onToggle={p => updateLab('wbc', 'performed', p)}
                    />
                  </div>
                  <div className="space-y-4">
                    <LabRow
                      label="Platelets"
                      unit="×10³/µL" refRange="150–450" refMin={150} refMax={450}
                      value={labs.platelets.value} performed={labs.platelets.performed}
                      min={1} max={1500} step={1}
                      onValueChange={v => updateLab('platelets', 'value', v)}
                      onToggle={p => updateLab('platelets', 'performed', p)}
                    />
                    <LabRow
                      label="Creatinine"
                      unit="mg/dL" refRange="0.6–1.2" refMin={0.6} refMax={1.2}
                      value={labs.creatinine.value} performed={labs.creatinine.performed}
                      min={0.1} max={30} step={0.01}
                      onValueChange={v => updateLab('creatinine', 'value', v)}
                      onToggle={p => updateLab('creatinine', 'performed', p)}
                    />
                    <LabRow
                      label="Serum CRP"
                      unit="mg/L" refRange="< 10.0" refMin={0} refMax={10.0}
                      value={labs.crp.value} performed={labs.crp.performed}
                      min={0.1} max={500} step={0.1}
                      onValueChange={v => updateLab('crp', 'value', v)}
                      onToggle={p => updateLab('crp', 'performed', p)}
                    />
                  </div>
                  <div className="space-y-4">
                    <LabRow
                      label="Bilirubin"
                      unit="mg/dL" refRange="< 1.2" refMin={0} refMax={1.2}
                      value={labs.bilirubin.value} performed={labs.bilirubin.performed}
                      min={0.1} max={30} step={0.1}
                      onValueChange={v => updateLab('bilirubin', 'value', v)}
                      onToggle={p => updateLab('bilirubin', 'performed', p)}
                    />
                    <div className="grid grid-cols-2 gap-4">
                      <LabRow
                        label="Dengue NS1"
                        unit="" refRange="Neg" refMin={0} refMax={0}
                        value={labs.dengueNS1.value} performed={labs.dengueNS1.performed}
                        min={0} max={1} step={1}
                        type="select" options={[{ l: 'Negative', v: 0 }, { l: 'Positive', v: 1 }]}
                        onValueChange={v => updateLab('dengueNS1', 'value', v)}
                        onToggle={p => updateLab('dengueNS1', 'performed', p)}
                      />
                      <LabRow
                        label="Malaria RDT"
                        unit="" refRange="Neg" refMin={0} refMax={0}
                        value={labs.malariaRDT.value} performed={labs.malariaRDT.performed}
                        min={0} max={1} step={1}
                        type="select" options={[{ l: 'Negative', v: 0 }, { l: 'Positive', v: 1 }]}
                        onValueChange={v => updateLab('malariaRDT', 'value', v)}
                        onToggle={p => updateLab('malariaRDT', 'performed', p)}
                      />
                    </div>
                  </div>
                </div>
              </section>

              {/* Section 3: AI Result Alert */}
              <AnimatePresence mode="wait">
                {result && (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col gap-3"
                  >
                    {/* Data Quality Warning Banner — shown when labs are absent/partial */}
                    {result.dataQualityWarnings?.length > 0 && (
                      <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 flex flex-col gap-1.5">
                        <div className="flex items-center gap-2">
                          <span className="text-amber-600 font-bold text-[12px] uppercase tracking-wide">⚠ Reduced Prediction Confidence</span>
                          <span className="ml-auto text-[11px] font-bold text-amber-700 bg-amber-100 border border-amber-300 rounded px-2 py-0.5">
                            Confidence {Math.round(result.confidenceScore * 100)}%
                          </span>
                        </div>
                        {result.dataQualityWarnings.map((w, i) => (
                          <p key={i} className="text-[12px] text-amber-800 leading-snug">{w}</p>
                        ))}
                      </div>
                    )}

                    <div className={`rounded-lg p-5 flex border-2 gap-5 items-center transition-all ${
                      result.alertLevel === 'critical' ? 'bg-[#fff5f5] border-brand-danger text-brand-text shadow-sm' :
                      result.alertLevel === 'warning' ? 'bg-orange-50 border-brand-warning text-brand-text' :
                      'bg-emerald-50 border-brand-success text-brand-text'
                    }`}
                    >
                    <div className={`text-center pr-5 border-r min-w-[100px] ${
                      result.alertLevel === 'critical' ? 'border-[#feb2b2]' :
                      result.alertLevel === 'warning' ? 'border-brand-warning/30' :
                      'border-brand-success/30'
                    }`}>
                      <div className={`text-3xl font-black ${
                        result.alertLevel === 'critical' ? 'text-brand-danger' :
                        result.alertLevel === 'warning' ? 'text-brand-warning' :
                        'text-brand-success'
                      }`}>
                        {result.aiScore.toFixed(2)}
                      </div>
                      <div className={`text-[10px] font-bold uppercase tracking-widest ${
                        result.alertLevel === 'critical' ? 'text-brand-danger' :
                        result.alertLevel === 'warning' ? 'text-brand-warning' :
                        'text-brand-success'
                      }`}>AI Probability</div>
                    </div>

                    <div className={`text-center pr-5 border-r min-w-[80px] ${
                      result.alertLevel === 'critical' ? 'border-[#feb2b2]' :
                      result.alertLevel === 'warning' ? 'border-brand-warning/30' :
                      'border-brand-success/30'
                    }`}>
                      <div className={`text-3xl font-black ${
                        result.alertLevel === 'critical' ? 'text-brand-danger' :
                        result.alertLevel === 'warning'  ? 'text-brand-warning' :
                        'text-brand-muted'
                      }`}>
                        {result.qsofaScore}<span className="text-xl opacity-40">/3</span>
                      </div>
                      <div className={`text-[10px] font-bold uppercase tracking-widest ${
                        result.alertLevel === 'critical' ? 'text-brand-danger' :
                        result.alertLevel === 'warning'  ? 'text-brand-warning' :
                        'text-brand-muted'
                      }`}>qSOFA Score</div>
                    </div>
                    
                    <div className="flex-1">
                      <div className={`font-black uppercase text-[15px] mb-1 ${
                        result.alertLevel === 'critical' ? 'text-brand-danger' :
                        result.alertLevel === 'warning' ? 'text-brand-warning' :
                        'text-brand-success'
                      }`}>
                        {result.alertLevel === 'critical' ? 'Critical Sepsis Alert' :
                        result.alertLevel === 'warning' ? 'Early Observation Required' :
                        'Patient Baseline Stable'}
                      </div>
                      <div className="text-[12px] opacity-90 leading-tight">
                        {result.alertLevel === 'critical' ? 'AI Model and Clinical qSOFA both confirm high risk. Immediate clinical intervention and broad-spectrum antibiotics recommended per protocol.' :
                        result.alertLevel === 'warning' ? 'High AI risk detected despite low clinical score. Observe for potential physiological compensation and monitoring.' :
                        'Patient risk markers remain within baseline parameters. Continue routine monitoring per unit protocol.'}
                      </div>
                    </div>
                    {/* Confidence meter row */}
                    {result.confidenceScore !== undefined && (
                      <div className="flex items-center gap-2 pt-1">
                        <span className="text-[10px] text-brand-text-secondary font-medium uppercase tracking-wide whitespace-nowrap">Model confidence</span>
                        <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all ${
                              result.confidenceScore >= 0.80 ? 'bg-brand-success' :
                              result.confidenceScore >= 0.60 ? 'bg-amber-400' : 'bg-red-400'
                            }`}
                            style={{ width: `${Math.round(result.confidenceScore * 100)}%` }}
                          />
                        </div>
                        <span className={`text-[10px] font-bold ${
                          result.confidenceScore >= 0.80 ? 'text-brand-success' :
                          result.confidenceScore >= 0.60 ? 'text-amber-500' : 'text-red-500'
                        }`}>{Math.round(result.confidenceScore * 100)}%</span>
                      </div>
                    )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Section 4: Clinical Explainability */}
              <section className="bg-white rounded-lg p-4 border border-brand-border shadow-[0_1px_3px_rgba(0,0,0,0.05)] space-y-4 shadow-sm">
                <div className="flex items-center justify-between border-b border-brand-border pb-2">
                  <h3 className="text-[13px] font-bold">AI Diagnostic Reasoning (SHAP & Attention)</h3>
                  <div className="flex items-center gap-1.5 px-2 py-0.5 bg-[#f8fafc] border border-brand-border rounded text-[10px] text-brand-muted font-bold">
                    <History size={10} />
                    Lookback: 6 Hours
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-brand-muted flex items-center justify-between">
                      <span>Feature Importance (SHAP Value)</span>
                      <span className="text-[9px] font-mono opacity-50">Local Influence</span>
                    </div>
                    <div className="bg-[#f8fafc] border border-brand-border rounded p-4 text-[11px] text-brand-muted h-[160px] flex flex-col items-center justify-center relative overflow-hidden">
                      <div className="absolute top-2 left-2 font-mono text-[8px] opacity-30 select-none">SHA_V1_INF</div>
                      {result?.shapDrivers?.length ? (
                        <div className="flex flex-col gap-2 w-full max-w-[240px]">
                          {result.shapDrivers.slice(0, 4).map(d => (
                            <ShapBar
                              key={d.feature}
                              label={d.feature.replace(/_/g, ' ')}
                              value={// FIX #27: width capped at 100 — shap*100 with reasonable scale
                        Math.min(100, Math.round(Math.abs(d.shap) * 150))}
                              color={d.shap > 0 ? 'var(--color-brand-danger)' : 'var(--color-brand-success)'}
                            />
                          ))}
                        </div>
                      ) : (
                        <div className="flex flex-col gap-2 w-full max-w-[240px] opacity-40">
                          <ShapBar label="Run analysis to load" value={0} color="var(--color-brand-muted)" />
                        </div>
                      )}
                      <div className="mt-auto pt-2 text-[9px] font-medium italic">
                        {result?.shapDrivers?.length
                          ? `Top driver: ${result.shapDrivers[0]?.feature.replace(/_/g, ' ')}`
                          : 'No analysis yet'}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-brand-muted flex items-center justify-between">
                      <span>Temporal Attention Heatmap</span>
                      <span className="text-[9px] font-mono opacity-50">TFT Sequence Analysis</span>
                    </div>
                    <div className="bg-[#f8fafc] border border-brand-border rounded p-3 h-[160px] flex flex-col items-center justify-center relative">
                      <div className="absolute top-2 left-2 font-mono text-[8px] opacity-30 select-none">TFT_ATT_SYNC</div>
                      {result ? (
                        <AttentionHeatmap weights={result.attentionWeights} />
                      ) : (
                        <div className="flex flex-col items-center gap-2 opacity-50">
                          <TrendingUp size={24} className="text-brand-muted" />
                          <p className="text-[10px] font-medium">Run analysis to generate temporal heatmap</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </section>
            </>
          ) : (
            <div className="flex flex-col gap-6">
              <section className="bg-white rounded-lg border border-brand-border shadow-[0_1px_3px_rgba(0,0,0,0.05)] p-4">
                <header className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-4">
                  <div>
                    <h3 className="text-sm font-bold text-brand-primary uppercase tracking-tight flex items-center gap-2">
                      <History size={16} />
                      Historical Record Analysis
                    </h3>
                    <p className="text-[11px] text-brand-muted">Retrospective physiological monitoring and laboratory patterns</p>
                  </div>
                  <div className="flex items-center gap-1 bg-[#f8fafc] p-1 border border-brand-border rounded-lg self-start">
                    {(['6h', '12h', '24h'] as const).map(range => (
                      <button
                        key={range}
                        onClick={() => setTimeRange(range)}
                        className={`px-3 py-1 text-[11px] font-bold rounded-md transition-all ${timeRange === range ? 'bg-brand-primary text-white shadow-sm' : 'text-brand-muted hover:bg-brand-primary/10'}`}
                      >
                        {range.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                  <div className="lg:col-span-2 space-y-6">
                    <h4 className="text-[11px] font-black uppercase text-brand-muted border-b border-brand-border pb-1">Vitals Trend Analysis</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
                      <HistoricalChartItem label="Heart Rate" data={vitalsHistory.hr} unit="BPM" color="var(--color-brand-primary)" />
                      <HistoricalChartItem label="Blood Pressure (MAP)" data={vitalsHistory.map} unit="mmHg" color="#10b981" />
                      <HistoricalChartItem label="Respiratory Rate" data={vitalsHistory.resp} unit="/min" color="#f59e0b" />
                      <HistoricalChartItem label="O2 Saturation" data={vitalsHistory.o2sat} unit="%" color="#3b82f6" />
                    </div>
                  </div>

                  <div className="space-y-6">
                    <h4 className="text-[11px] font-black uppercase text-brand-muted border-b border-brand-border pb-1">Laboratory Summary</h4>
                    <div className="bg-[#f8fafc] border border-brand-border rounded-lg overflow-hidden shadow-sm">
                      <table className="w-full text-left text-[12px]">
                        <thead className="bg-[#edf2f7] text-brand-muted font-bold text-[10px] uppercase">
                          <tr>
                            <th className="px-4 py-3">Parameter</th>
                            <th className="px-4 py-3">Value</th>
                            <th className="px-4 py-3">Status</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-brand-border bg-white">
                          <HistoricalLabRow label="Serum Lactate" value={labs.lactate.value} refRange="< 2.0" status={labs.lactate.value > 2.0 ? 'danger' : 'normal'} />
                          <HistoricalLabRow label="Serum CRP" value={labs.crp.value} refRange="< 10.0" status={labs.crp.value > 10.0 ? 'danger' : 'normal'} />
                          <HistoricalLabRow label="WBC Count" value={labs.wbc.value} refRange="4.0-11.0" status={(labs.wbc.value > 11.0 || labs.wbc.value < 4.0) ? 'warning' : 'normal'} />
                          <HistoricalLabRow label="Platelets" value={labs.platelets.value} refRange="150-450" status={labs.platelets.value < 150 ? 'warning' : 'normal'} />
                          <HistoricalLabRow label="Creatinine" value={labs.creatinine.value} refRange="0.6-1.2" status={labs.creatinine.value > 1.2 ? 'warning' : 'normal'} />
                          <HistoricalLabRow label="Bilirubin" value={labs.bilirubin.value} refRange="< 1.2" status={labs.bilirubin.value > 1.2 ? 'warning' : 'normal'} />
                        </tbody>
                      </table>
                    </div>
                    
                    <div className="p-4 bg-orange-50 border border-brand-warning/20 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Activity size={14} className="text-brand-warning" />
                        <h5 className="text-[11px] font-bold text-brand-warning uppercase">Clinical Note</h5>
                      </div>
                      {/* FIX #24: dynamic note derived from actual vitals, not hardcoded tachycardia */}
                      <p className="text-[11px] text-brand-text opacity-80 leading-snug">
                        {(() => {
                          const findings: string[] = [];
                          if (vitals.hr   > 100) findings.push(`tachycardia (HR ${Math.round(vitals.hr)} bpm)`);
                          if (vitals.resp > 20)  findings.push(`tachypnoea (RR ${Math.round(vitals.resp)} /min)`);
                          if (vitals.o2sat < 95) findings.push(`hypoxaemia (SpO₂ ${Math.round(vitals.o2sat)}%)`);
                          if (vitals.map  < 65)  findings.push(`hypotension (MAP ${Math.round(vitals.map)} mmHg)`);
                          return findings.length === 0
                            ? `Retrospective ${timeRange} window: vitals within normal limits. Continue routine monitoring.`
                            : `Retrospective ${timeRange} window indicates ${findings.join(', ')}. Close monitoring required.`;
                        })()}
                      </p>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          )}
        </div>
      </main>
        {/* ── Patient Picker Modal ───────────────────────────────── */}
        <AnimatePresence>
          {showPatientPicker && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4"
              onClick={() => setShowPatientPicker(false)}
            >
              <motion.div
                initial={{ scale: 0.95, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.95, opacity: 0 }}
                className="bg-white rounded-xl shadow-2xl w-full max-w-lg p-6"
                onClick={e => e.stopPropagation()}
              >
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h2 className="text-[16px] font-semibold text-brand-text">Load Patient File</h2>
                    <p className="text-[12px] text-brand-muted mt-0.5">
                      Select a patient data file. The backend will stream vitals to the dashboard.
                    </p>
                  </div>
                  <button onClick={() => setShowPatientPicker(false)}
                    className="p-1.5 hover:bg-gray-100 rounded text-brand-muted">✕</button>
                </div>

                {patients.length === 0 ? (
                  <div className="py-8 text-center text-brand-muted text-[13px]">
                    <Database size={32} className="mx-auto mb-2 opacity-30" />
                    No patient files found in backend/data/patients/
                  </div>
                ) : (
                  <div className="flex flex-col gap-2 max-h-80 overflow-y-auto">
                    {patients.map(p => (
                      <button key={p.patient_id}
                        onClick={() => startFileFeed(p.patient_id)}
                        className="text-left p-4 rounded-lg border border-brand-border hover:border-blue-400 hover:bg-blue-50 transition-all group"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1 min-w-0">
                            <p className="text-[13px] font-semibold text-brand-text group-hover:text-blue-700">
                              {p.label}
                            </p>
                            <p className="text-[11px] text-brand-muted mt-0.5 leading-snug line-clamp-2">
                              {p.description}
                            </p>
                          </div>
                          <div className="flex flex-col items-end gap-1 shrink-0">
                            <span className="text-[10px] font-medium bg-gray-100 text-gray-500 px-2 py-0.5 rounded">
                              {p.total_readings} readings
                            </span>
                            {p.has_labs && (
                              <span className="text-[10px] font-medium bg-green-100 text-green-700 px-2 py-0.5 rounded">
                                + Labs
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-1 mt-2">
                          <PlayCircle size={12} className="text-blue-400 group-hover:text-blue-600" />
                          <span className="text-[11px] text-blue-500 group-hover:text-blue-700">
                            Stream at {p.interval_minutes}min intervals
                          </span>
                        </div>
                      </button>
                    ))}
                  </div>
                )}

                <p className="text-[10px] text-brand-muted mt-4 pt-3 border-t border-brand-border">
                  Add custom patients by dropping JSON files into <code className="bg-gray-100 px-1 rounded">backend/data/patients/</code>
                </p>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
    </div>
  );
}

// --- Specific Dashboard Components ---

function HistoricalChartItem({ label, data, unit, color }: { label: string, data: number[], unit: string, color: string }) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center px-1">
        <span className="text-[11px] font-bold text-brand-text/80">{label}</span>
        <span className="text-[10px] font-mono text-brand-muted font-bold">{data[data.length - 1]?.toFixed(1)} {unit}</span>
      </div>
      <div className="bg-[#fcfcfc] border border-brand-border rounded-lg p-2 h-[80px] shadow-sm">
        <DetailedChart 
          data={data} 
          width={280} 
          height={60} 
          color={color} 
          minVal={Math.min(...data) - 5} 
          maxVal={Math.max(...data) + 5} 
        />
      </div>
    </div>
  );
}

function HistoricalLabRow({ label, value, refRange, status }: { label: string, value: number, refRange: string, status: 'normal' | 'warning' | 'danger' }) {
  return (
    <tr className="hover:bg-[#f8fafc] transition-colors">
      <td className="px-4 py-3">
        <div className="font-bold text-brand-text">{label}</div>
        <div className="text-[9px] text-brand-muted font-bold opacity-60">Ref: {refRange}</div>
      </td>
      <td className="px-4 py-3 font-mono text-brand-text font-bold">{value.toFixed(2)}</td>
      <td className="px-4 py-3">
        <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase ${
          status === 'danger' ? 'bg-red-50 text-red-600 border border-red-100' : 
          status === 'warning' ? 'bg-orange-50 text-orange-600 border border-orange-100' : 
          'bg-emerald-50 text-emerald-600 border border-emerald-100'
        }`}>
          {status}
        </span>
      </td>
    </tr>
  );
}

function ShapBar({ label, value, color }: { label: string, value: number, color: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[9px] font-bold uppercase">
        <span>{label}</span>
        <span>{value}%</span>
      </div>
      <div className="h-1.5 bg-[#edf2f7] rounded-full overflow-hidden">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          style={{ backgroundColor: color }}
          className="h-full"
        />
      </div>
    </div>
  );
}

function AttentionHeatmap({ weights }: { weights: number[] }) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !weights.length) return;

    const svg = d3.select(svgRef.current);
    // BUG FIX: always clear before drawing to prevent D3 double-render in
    // React 19 StrictMode (which double-invokes effects on mount).
    svg.selectAll("*").remove();

    const width = 280;
    const height = 90;
    const padding = 25;

    const x = d3.scaleBand()
      .domain(['H-5', 'H-4', 'H-3', 'H-2', 'H-1', 'H-0'])
      .range([padding, width - padding])
      .padding(0.1);

    const colorScale = d3.scaleSequential()
      .domain([0, d3.max(weights) || 1])
      .interpolator(d3.interpolateBlues);

    // Cells
    svg.selectAll("rect")
      .data(weights)
      .enter()
      .append("rect")
      .attr("x", (_, i) => x(`H-${5-i}`) || 0)
      .attr("y", 10)
      .attr("width", x.bandwidth())
      .attr("height", 45)
      .attr("rx", 4)
      .attr("fill", d => colorScale(d))
      .attr("stroke", "#ffffff")
      .attr("stroke-width", 1)
      .append("title")
      .text(d => `Weight: ${d.toFixed(3)}`);

    // Labels
    svg.selectAll("text.label")
      .data(['H-5', 'H-4', 'H-3', 'H-2', 'H-1', 'H-0'])
      .enter()
      .append("text")
      .attr("class", "label")
      .attr("x", d => (x(d) || 0) + x.bandwidth() / 2)
      .attr("y", 70)
      .attr("text-anchor", "middle")
      .attr("font-size", "9px")
      .attr("font-weight", "bold")
      .attr("fill", "#2D3748")
      .text(d => d);

    // Value labels
    svg.selectAll("text.value")
      .data(weights)
      .enter()
      .append("text")
      .attr("class", "value")
      .attr("x", (_, i) => (x(`H-${5-i}`) || 0) + x.bandwidth() / 2)
      .attr("y", 38)
      .attr("text-anchor", "middle")
      .attr("font-size", "8px")
      .attr("font-weight", "bold")
      .attr("fill", d => d > 0.2 ? "white" : "#2D3748")
      .text(d => d.toFixed(2));

    // BUG FIX: cleanup function removes all SVG nodes on unmount / re-render
    // to prevent stale element accumulation on hot-reload
    return () => {
      d3.select(svgRef.current).selectAll("*").remove();
    };
  }, [weights]);

  return <svg ref={svgRef} width={280} height={90} className="overflow-visible" />;
}

// --- Helper Components ---

function VitalInput({ label, value, history = [], min, max, step = 1, onChange }: { 
  label: string, value: number, history?: number[], min: number, max: number, step?: number, onChange: (v: number) => void 
}) {
  const [showDelta, setShowDelta] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  // Local string state prevents glitching when typing negatives or decimals mid-entry
  const [inputStr, setInputStr] = useState(value.toFixed(step < 1 ? 1 : 0));

  // Keep local string in sync when value changes externally (e.g. simulation)
  useEffect(() => {
    setInputStr(value.toFixed(step < 1 ? 1 : 0));
  }, [value, step]);

  const delta = useMemo(() => {
    if (history.length < 2) return 0;
    return value - history[0];
  }, [value, history]);

  const isPositive = delta > 0;
  const absDelta = Math.abs(delta).toFixed(step < 1 ? 1 : 0);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setInputStr(raw);
    const parsed = parseFloat(raw);
    if (!isNaN(parsed)) {
      onChange(Math.min(max, Math.max(min, parsed)));
    }
  };

  const handleBlur = () => {
    // On blur, snap display back to the committed value
    setInputStr(value.toFixed(step < 1 ? 1 : 0));
  };

  return (
    <div className="vital-input flex flex-col gap-1 relative">
      <div className="flex justify-between items-center h-4">
        <span className="text-[10px] text-brand-muted uppercase font-bold">{label}</span>
        <button 
          onClick={() => setShowDelta(!showDelta)}
          className={`text-[9px] font-bold px-1.5 py-0.5 rounded transition-colors flex items-center gap-0.5 ${
            showDelta ? 'bg-brand-primary text-white' : 'text-brand-muted hover:bg-brand-primary/10'
          }`}
          title="Toggle 3h Trend Delta"
        >
          {showDelta ? 'Δ' : '...'}
        </button>
      </div>
      <div 
        className="relative group focus-within:z-50"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <input 
          type="number" 
          step={step}
          min={min}
          max={max}
          value={inputStr}
          onChange={handleChange}
          onBlur={handleBlur}
          className="w-full bg-[#fafafa] border border-brand-border rounded p-2 pr-12 text-[13px] outline-none hover:bg-white focus:bg-white focus:border-brand-primary font-mono transition-all"
        />
        
        <AnimatePresence>
          {showDelta && (
            <motion.div 
              initial={{ opacity: 0, x: 5 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 5 }}
              className={`absolute -right-12 top-1/2 -translate-y-1/2 px-1.5 py-1 rounded shadow-sm text-[10px] font-bold border flex items-center gap-1 z-20 ${
                delta === 0 ? 'bg-slate-50 text-slate-500 border-slate-200' :
                isPositive ? 'bg-red-50 text-red-600 border-red-100' : 
                'bg-emerald-50 text-emerald-600 border-emerald-100'
              }`}
            >
              {delta !== 0 && (isPositive ? <TrendingUp size={10} /> : <TrendingDown size={10} />)}
              {isPositive ? '+' : delta < 0 ? '-' : ''}{absDelta}
              <span className="text-[8px] opacity-60">3h</span>
            </motion.div>
          )}

          {isHovered && history.length > 0 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: -10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -10 }}
              className="absolute left-0 bottom-full mb-2 w-[240px] bg-white border border-brand-border rounded-lg shadow-xl p-3 z-[100] pointer-events-none"
            >
              <div className="text-[10px] font-bold uppercase text-brand-muted mb-2 flex justify-between">
                <span>6-Hour Time Trend</span>
                <span className="text-brand-primary">{value.toFixed(step < 1 ? 1 : 0)} {label.split('(')[1]?.split(')')[0] || ''}</span>
              </div>
              <DetailedChart 
                data={history} 
                width={214} 
                height={100} 
                color="var(--color-brand-primary)" 
                minVal={min}
                maxVal={max}
              />
              <div className="flex justify-between mt-2 text-[8px] font-bold text-brand-muted opacity-60">
                <span>-6 HR</span>
                <span>-3 HR</span>
                <span>NOW</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="absolute right-1 top-1 bottom-1 w-10 flex items-center justify-center bg-white/50 pointer-events-none rounded">
          <Sparkline data={history} width={36} height={16} color="var(--color-brand-primary)" />
        </div>
      </div>
    </div>
  );
}

function DetailedChart({ data, width, height, color, minVal, maxVal }: { 
  data: number[], width: number, height: number, color: string, minVal: number, maxVal: number 
}) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 5, right: 5, bottom: 20, left: 25 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear()
      .domain([0, data.length - 1])
      .range([0, innerWidth]);

    const y = d3.scaleLinear()
      .domain([d3.min(data) || minVal, d3.max(data) || maxVal])
      .nice()
      .range([innerHeight, 0]);

    // Grid lines
    g.append("g")
      .attr("class", "grid")
      .attr("opacity", 0.05)
      .call(d3.axisLeft(y).ticks(4).tickSize(-innerWidth).tickFormat(() => ""));

    // Axes
    const yAxis = d3.axisLeft(y).ticks(4).tickSize(0).tickPadding(5);
    g.append("g")
      .attr("class", "y-axis")
      .attr("font-size", "8px")
      .attr("font-weight", "bold")
      .attr("color", "#718096")
      .call(yAxis)
      .select(".domain").remove();

    const line = d3.line<number>()
      .x((_, i) => x(i))
      .y(d => y(d))
      .curve(d3.curveMonotoneX);

    // Area
    const area = d3.area<number>()
      .x((_, i) => x(i))
      .y0(innerHeight)
      .y1(d => y(d))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(data)
      .attr("fill", color)
      .attr("opacity", 0.05)
      .attr("d", area);

    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 2)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("d", line);

    // Points
    g.selectAll("circle")
      .data(data)
      .enter()
      .append("circle")
      .attr("cx", (_, i) => x(i))
      .attr("cy", d => y(d))
      .attr("r", 2)
      .attr("fill", "#fff")
      .attr("stroke", color)
      .attr("stroke-width", 1);

    // BUG FIX: cleanup to prevent double-render accumulation in React 19 StrictMode
    return () => {
      d3.select(svgRef.current).selectAll("*").remove();
    };
  }, [data, width, height, color, minVal, maxVal]);

  return <svg ref={svgRef} width={width} height={height} className="overflow-visible" />;
}

function Sparkline({ data, width, height, color }: { data: number[], width: number, height: number, color: string }) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const x = d3.scaleLinear()
      .domain([0, data.length - 1])
      .range([2, width - 2]);

    const y = d3.scaleLinear()
      .domain([d3.min(data) || 0, d3.max(data) || 1])
      .range([height - 2, 2]);

    const line = d3.line<number>()
      .x((_, i) => x(i))
      .y(d => y(d))
      .curve(d3.curveBasis);

    svg.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 1.5)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("d", line);

    // BUG FIX: cleanup to prevent double-render accumulation
    return () => {
      d3.select(svgRef.current).selectAll("*").remove();
    };
  }, [data, width, height, color]);

  return <svg ref={svgRef} width={width} height={height} className="overflow-visible" />;
}

function LabRow({
  label, unit = '', refRange = '', refMin, refMax,
  value, performed, min = 0, max = 9999, step = 0.1,
  onValueChange, onToggle,
  type = 'number', options = []
}: {
  label: string;
  unit?: string;
  refRange?: string;
  refMin?: number;
  refMax?: number;
  value: number;
  performed: boolean;
  min?: number;
  max?: number;
  step?: number;
  onValueChange: (v: number) => void;
  onToggle: (p: boolean) => void;
  type?: 'number' | 'select';
  options?: { l: string; v: number }[];
}) {
  // Local string state: allows partial entry (e.g. "0.", "12.") without snapping
  const [inputStr, setInputStr] = useState(String(value));
  const [focused, setFocused] = useState(false);

  // Sync external value changes (e.g. reset) when not focused
  useEffect(() => {
    if (!focused) setInputStr(String(value));
  }, [value, focused]);

  const isAbnormal = performed && type === 'number' && refMin !== undefined && refMax !== undefined &&
    (value < refMin || value > refMax);

  // Severity: critical if > 2× upper or < 0.5× lower
  const isCritical = performed && isAbnormal && refMax !== undefined && refMin !== undefined &&
    (value > refMax * 2 || (refMin > 0 && value < refMin * 0.5));

  const borderClass = !performed
    ? 'border-brand-border'
    : isCritical
    ? 'border-red-400 ring-1 ring-red-200'
    : isAbnormal
    ? 'border-orange-400 ring-1 ring-orange-100'
    : 'border-emerald-400';

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setInputStr(raw);
    const parsed = parseFloat(raw);
    if (!isNaN(parsed) && parsed >= min && parsed <= max) {
      onValueChange(parsed);
    }
  };

  const handleBlur = () => {
    setFocused(false);
    // Clamp and commit on blur
    const parsed = parseFloat(inputStr);
    if (isNaN(parsed)) {
      setInputStr(String(value)); // revert to last good value
    } else {
      const clamped = Math.min(max, Math.max(min, parsed));
      onValueChange(clamped);
      setInputStr(String(clamped));
    }
  };

  return (
    <div className={`flex flex-col gap-1.5 transition-opacity ${!performed && 'opacity-55'}`}>
      {/* Header row: label + unit + ref range + toggle */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-baseline gap-1 min-w-0">
          <span className="text-[11px] font-bold text-brand-text truncate">{label}</span>
          {unit && <span className="text-[9px] text-brand-muted font-bold shrink-0">{unit}</span>}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {refRange && (
            <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${
              isCritical ? 'bg-red-50 text-red-500 border-red-200' :
              isAbnormal ? 'bg-orange-50 text-orange-500 border-orange-200' :
              'bg-slate-50 text-brand-muted border-brand-border'
            }`}>
              Ref: {refRange}
            </span>
          )}
          {/* Styled toggle switch */}
          <button
            onClick={() => onToggle(!performed)}
            className={`relative inline-flex w-9 h-5 rounded-full transition-colors focus:outline-none shrink-0 ${
              performed ? 'bg-brand-primary' : 'bg-slate-200'
            }`}
            title={performed ? 'Mark as not performed' : 'Mark as performed'}
          >
            <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-transform ${
              performed ? 'translate-x-4' : 'translate-x-0'
            }`} />
          </button>
        </div>
      </div>

      {/* Input row */}
      {type === 'select' ? (
        <div className="relative">
          <select
            disabled={!performed}
            value={value}
            onChange={e => onValueChange(parseInt(e.target.value))}
            className={`w-full bg-[#fafafa] border rounded px-2.5 py-1.5 text-[12px] font-medium outline-none appearance-none cursor-pointer transition-all
              disabled:bg-slate-50 disabled:text-brand-muted disabled:cursor-default
              focus:bg-white ${borderClass}`}
          >
            {options.map(opt => <option key={opt.v} value={opt.v}>{opt.l}</option>)}
          </select>
          <div className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-brand-muted">
            <svg width="10" height="6" viewBox="0 0 10 6" fill="none">
              <path d="M1 1l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        </div>
      ) : (
        <div className="relative">
          <input
            disabled={!performed}
            type="number"
            inputMode="decimal"
            step={step}
            min={min}
            max={max}
            value={focused ? inputStr : (performed ? value : '')}
            placeholder={performed ? `${min}–${max}` : 'Not performed'}
            onChange={handleChange}
            onFocus={() => { setFocused(true); setInputStr(String(value)); }}
            onBlur={handleBlur}
            className={`w-full bg-[#fafafa] border rounded px-2.5 py-1.5 pr-16 text-[12px] font-mono outline-none transition-all
              disabled:bg-slate-50 disabled:text-brand-muted disabled:cursor-default
              focus:bg-white ${borderClass}`}
          />
          {/* Status chip inside input */}
          {performed && (
            <span className={`absolute right-2 top-1/2 -translate-y-1/2 text-[8px] font-bold px-1 py-0.5 rounded ${
              isCritical ? 'bg-red-100 text-red-600' :
              isAbnormal ? 'bg-orange-100 text-orange-600' :
              'bg-emerald-100 text-emerald-700'
            }`}>
              {isCritical ? 'CRIT' : isAbnormal ? 'ABN' : 'NML'}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function RiskToggle({ label, active, onClick }: { label: string, active: boolean, onClick: () => void }) {
  return (
    <button 
      onClick={onClick}
      className={`flex items-center justify-between px-3 py-2 rounded border transition-all text-left ${active ? 'bg-indigo-50 border-indigo-200 text-indigo-700 shadow-sm' : 'bg-[#fafafa] border-brand-border text-brand-muted hover:border-brand-muted/30'}`}
    >
      <span className="text-[11px] font-bold uppercase tracking-tight">{label}</span>
      <div className={`w-2 h-2 rounded-full ${active ? 'bg-indigo-500 animate-pulse' : 'bg-slate-200'}`} />
    </button>
  );
}
