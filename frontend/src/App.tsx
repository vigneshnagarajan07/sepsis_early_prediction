/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useMemo, useEffect, useRef } from 'react';
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
  History
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
  priorAntibiotics: boolean; // Critical Risk Factor
  referredFromOutside: boolean; // Critical Risk Factor
  gramNegativeRisk: boolean;
  malariaEndemic: boolean;
  dengueEndemic: boolean;
  covidPrevalence: boolean;
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
  dengueNS1: LabTest;
  malariaRDT: LabTest;
}

// --- Components ---

export default function SepsisDashboard() {
  // 1. Patient Demographics (Sidebar)
  const [demographics, setDemographics] = useState<PatientDemographics>({
    age: 45,
    gender: 'Male',
    bmi: 24.5,
    diabetes: false,
    ckd: false,
    priorAntibiotics: false,
    referredFromOutside: false,
    gramNegativeRisk: false,
    malariaEndemic: false,
    dengueEndemic: false,
    covidPrevalence: false,
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
  const [labs, setLabs] = useState<Labs>({
    lactate: { value: 1.2, performed: true },
    pct: { value: 0.05, performed: true },
    wbc: { value: 8.5, performed: true },
    platelets: { value: 250, performed: true },
    creatinine: { value: 1.0, performed: true },
    bilirubin: { value: 0.5, performed: true },
    dengueNS1: { value: 0, performed: true },
    malariaRDT: { value: 0, performed: true },
  });

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [viewMode, setViewMode] = useState<'live' | 'historical'>('live');
  const [timeRange, setTimeRange] = useState<'6h' | '12h' | '24h'>('6h');
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [result, setResult] = useState<{
    aiScore: number;
    qsofaScore: number;
    alertLevel: 'none' | 'warning' | 'critical';
    attentionWeights: number[];
  } | null>(null);

  // --- Logic ---

  const calculateQsofa = (v: Vitals) => {
    let score = 0;
    if (v.resp >= 22) score += 1;
    if (v.gcs < 15) score += 1;
    if (v.systolicBp <= 100) score += 1;
    return score;
  };

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    setResult(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vitals,
          labs,
          demographics
        })
      });

      if (!response.ok) {
        let errorMsg = `Prediction failed (${response.status})`;
        try {
          const errorJson = await response.json();
          errorMsg += `: ${JSON.stringify(errorJson)}`;
        } catch {
          const errorText = await response.text();
          errorMsg += `: ${errorText.substring(0, 100)}`;
        }
        throw new Error(errorMsg);
      }
      const analysis = await response.json();
      setResult(analysis);
    } catch (error) {
      console.error("Inference Error:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

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
  // 1. Data Drift (Vitals Only)
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isSimulating) {
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
      }, 5000); // Updated to 5 seconds
    }
    return () => clearInterval(interval);
  }, [isSimulating]);

  // 2. Auto-Analysis sync
  useEffect(() => {
    if (isSimulating) {
      runAnalysis();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vitals, labs, demographics, isSimulating]);

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
              <input type="text" value="P-99201" readOnly className="w-full bg-[#fafafa] border border-brand-border rounded px-2 py-1.5 text-[13px] outline-none" />
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
                    <RiskToggle 
                      label="COVID-19 Prevalence" 
                      active={demographics.covidPrevalence} 
                      onClick={() => setDemographics({...demographics, covidPrevalence: !demographics.covidPrevalence})} 
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
              {isSimulating ? <EyeOff size={14} /> : <Eye size={14} />}
              {isSimulating ? 'Stop Live Feed' : 'Start Live Feed'}
            </button>
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
                      <span className="text-[10px] text-brand-muted uppercase font-bold">Delta HR (3h)</span>
                      <input type="number" value={12} readOnly className="w-full bg-[#edf2f7] border border-brand-border rounded p-2 text-[13px] outline-none" />
                    </div>
                  </div>

                  <div className="mt-5 pt-4 border-t border-brand-border flex justify-end">
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

                <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-3">
                  <div className="space-y-3">
                    <LabRow label="Serum Lactate" value={labs.lactate.value} performed={labs.lactate.performed} onValueChange={v => updateLab('lactate', 'value', v)} onToggle={p => updateLab('lactate', 'performed', p)} />
                    <LabRow label="Procalcitonin" value={labs.pct.value} performed={labs.pct.performed} onValueChange={v => updateLab('pct', 'value', v)} onToggle={p => updateLab('pct', 'performed', p)} />
                    <LabRow label="WBC Count" value={labs.wbc.value} performed={labs.wbc.performed} onValueChange={v => updateLab('wbc', 'value', v)} onToggle={p => updateLab('wbc', 'performed', p)} />
                  </div>
                  <div className="space-y-3">
                    <LabRow label="Platelets" value={labs.platelets.value} performed={labs.platelets.performed} onValueChange={v => updateLab('platelets', 'value', v)} onToggle={p => updateLab('platelets', 'performed', p)} />
                    <LabRow label="Creatinine" value={labs.creatinine.value} performed={labs.creatinine.performed} onValueChange={v => updateLab('creatinine', 'value', v)} onToggle={p => updateLab('creatinine', 'performed', p)} />
                  </div>
                  <div className="space-y-3">
                    <LabRow label="Bilirubin" value={labs.bilirubin.value} performed={labs.bilirubin.performed} onValueChange={v => updateLab('bilirubin', 'value', v)} onToggle={p => updateLab('bilirubin', 'performed', p)} />
                    <div className="grid grid-cols-2 gap-4">
                      <LabRow label="Dengue" value={labs.dengueNS1.value} performed={labs.dengueNS1.performed} onValueChange={v => updateLab('dengueNS1', 'value', v)} type="select" options={[{ l: 'Neg', v: 0 }, { l: 'Pos', v: 1 }]} onToggle={p => updateLab('dengueNS1', 'performed', p)} />
                      <LabRow label="Malaria" value={labs.malariaRDT.value} performed={labs.malariaRDT.performed} onValueChange={v => updateLab('malariaRDT', 'value', v)} type="select" options={[{ l: 'Neg', v: 0 }, { l: 'Pos', v: 1 }]} onToggle={p => updateLab('malariaRDT', 'performed', p)} />
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
                    className={`rounded-lg p-5 flex border-2 gap-5 items-center transition-all ${
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
                        result.alertLevel === 'text-brand-text'
                      }`}>
                        {result.qsofaScore}<span className="text-xl opacity-40">/3</span>
                      </div>
                      <div className={`text-[10px] font-bold uppercase tracking-widest ${
                        result.alertLevel === 'critical' ? 'text-brand-danger' :
                        result.alertLevel === 'text-brand-muted'
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
                      <div className="flex flex-col gap-2 w-full max-w-[240px]">
                        <ShapBar label="Prior Antibiotics" value={85} color="var(--color-brand-danger)" />
                        <ShapBar label="Serum Lactate" value={72} color="var(--color-brand-warning)" />
                        <ShapBar label="Resp Rate" value={64} color="var(--color-brand-warning)" />
                        <ShapBar label="Heart Rate" value={45} color="var(--color-brand-primary)" />
                      </div>
                      <div className="mt-auto pt-2 text-[9px] font-medium italic">High Influence: Prior Antibiotics, Lactate</div>
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
                      <p className="text-[11px] text-brand-text opacity-80 leading-snug">
                        Retrospective analysis of the historical {timeRange} window indicates persistent tachycardia. Continuous monitoring required.
                      </p>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          )}
        </div>
      </main>
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

    // Labelling
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

    // Value Labels
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

  }, [weights]);

  return <svg ref={svgRef} width={280} height={90} className="overflow-visible" />;
}

// --- Helper Components ---

function VitalInput({ label, value, history = [], min, max, step = 1, onChange }: { 
  label: string, value: number, history?: number[], min: number, max: number, step?: number, onChange: (v: number) => void 
}) {
  const [showDelta, setShowDelta] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  
  const delta = useMemo(() => {
    if (history.length < 2) return 0;
    return value - history[0];
  }, [value, history]);

  const isPositive = delta > 0;
  const absDelta = Math.abs(delta).toFixed(step < 1 ? 1 : 0);

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
          value={value.toFixed(step < 1 ? 1 : 0)}
          onChange={e => onChange(parseFloat(e.target.value) || 0)}
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

  }, [data, width, height, color]);

  return <svg ref={svgRef} width={width} height={height} className="overflow-visible" />;
}

function LabRow({ label, value, performed, onValueChange, onToggle, type = 'number', options = [] }: {
  label: string, value: number, performed: boolean, onValueChange: (v: number) => void, onToggle: (p: boolean) => void, type?: 'number' | 'select', options?: { l: string, v: number }[]
}) {
  return (
    <div className={`grid grid-cols-[1fr_80px_40px] items-center gap-2 group transition-opacity ${!performed && 'opacity-60'}`}>
      <label className="text-[11px] font-medium text-brand-text truncate">{label}</label>
      
      {type === 'select' ? (
        <select 
          disabled={!performed}
          value={value}
          onChange={e => onValueChange(parseInt(e.target.value))}
          className="bg-[#fafafa] border border-brand-border rounded px-1.5 py-1 text-[12px] outline-none disabled:bg-slate-100/50"
        >
          {options.map(opt => <option key={opt.v} value={opt.v}>{opt.l}</option>)}
        </select>
      ) : (
        <input 
          disabled={!performed}
          type="number" 
          step="0.1"
          value={performed ? value.toFixed(1) : -9.9}
          onChange={e => onValueChange(parseFloat(e.target.value) || 0)}
          className={`bg-[#fafafa] border border-brand-border rounded px-1.5 py-1 text-[12px] font-mono outline-none disabled:bg-slate-100/50`}
        />
      )}

      <div className="flex justify-end">
        <input 
          type="checkbox" 
          checked={performed} 
          onChange={e => onToggle(e.target.checked)}
          className="w-4 h-4 rounded border-brand-border text-brand-primary focus:ring-brand-primary/20 cursor-pointer" 
        />
      </div>
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
