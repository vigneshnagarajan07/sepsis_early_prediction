/**
 * mockPatientFeed.js
 *
 * Built-in demo patient — a 58-year-old diabetic male referred from an
 * outside facility, showing classic sepsis progression over 12 readings.
 * Each reading is replayed at 5-second intervals (demo speed).
 * In real clinical use this would be 15-minute intervals; the demo
 * compresses 3 hours of physiology into 60 seconds.
 *
 * Vitals trajectory:
 *   T+0  : Mild tachycardia, normal MAP — patient appears stable
 *   T+15m: HR climbing, first temp spike
 *   T+30m: MAP beginning to fall, RR escalating
 *   T+45m: Labs drawn — elevated WBC, lactate 1.9 (borderline)
 *   T+60m: Clear hemodynamic instability — MAP < 70
 *   T+75m: Lactate 2.4, PCT elevated — sepsis criteria met
 *   T+90m: GCS beginning to drop — end-organ involvement
 *   T+105m: MAP < 65 (septic shock threshold), escalation required
 *   T+120m: Critical — lactate 3.2, GCS 12, urine dropping
 *   T+135m: Full septic shock picture
 *   T+150m: Post-intervention (slight stabilisation)
 *   T+165m: Continued monitoring
 */

const _noLab = (val) => ({ value: val, performed: false });
const _lab   = (val) => ({ value: val, performed: true  });

export const mockPatientFeed = {
  /** Demo interval — 5 seconds per reading (clinical: 900 s = 15 min) */
  intervalSeconds: 5,

  demographics: {
    age: 58,
    gender: 'Male',
    bmi: 26.2,
    diabetes: true,
    ckd: false,
    cirrhosis: false,
    malignancy: false,
    immunosuppression: false,
    priorAntibiotics: true,       // ← severe risk escalation
    referredFromOutside: true,    // ← severe risk escalation
    gramNegativeRisk: true,
    malariaEndemic: false,
    dengueEndemic: false,
  },

  readings: [
    // ── T+0min  Baseline admission ──────────────────────────────────────────
    {
      vitals: {
        hr: 94, map: 82, resp: 19, temp: 37.8,
        o2sat: 97, urineOutput: 0.68, gcs: 15, systolicBp: 118,
      },
      labs: {
        lactate:    _noLab(1.4),
        pct:        _noLab(0.08),
        wbc:        _noLab(10.2),
        platelets:  _noLab(218),
        creatinine: _noLab(1.1),
        bilirubin:  _noLab(0.6),
        crp:        _noLab(14),
        dengueNS1:  _noLab(0),
        malariaRDT: _noLab(0),
      },
    },

    // ── T+15min  HR climbing, temperature rises ──────────────────────────────
    {
      vitals: {
        hr: 102, map: 79, resp: 21, temp: 38.3,
        o2sat: 97, urineOutput: 0.62, gcs: 15, systolicBp: 115,
      },
      labs: {
        lactate:    _noLab(1.4),
        pct:        _noLab(0.08),
        wbc:        _noLab(10.2),
        platelets:  _noLab(218),
        creatinine: _noLab(1.1),
        bilirubin:  _noLab(0.6),
        crp:        _noLab(14),
        dengueNS1:  _noLab(0),
        malariaRDT: _noLab(0),
      },
    },

    // ── T+30min  MAP falling, RR escalating ─────────────────────────────────
    {
      vitals: {
        hr: 109, map: 75, resp: 23, temp: 38.7,
        o2sat: 96, urineOutput: 0.55, gcs: 15, systolicBp: 110,
      },
      labs: {
        lactate:    _noLab(1.4),
        pct:        _noLab(0.08),
        wbc:        _noLab(10.2),
        platelets:  _noLab(218),
        creatinine: _noLab(1.1),
        bilirubin:  _noLab(0.6),
        crp:        _noLab(14),
        dengueNS1:  _noLab(0),
        malariaRDT: _noLab(0),
      },
    },

    // ── T+45min  FIRST LABS DRAWN — borderline sepsis ───────────────────────
    {
      vitals: {
        hr: 114, map: 72, resp: 24, temp: 38.9,
        o2sat: 95, urineOutput: 0.50, gcs: 15, systolicBp: 106,
      },
      labs: {
        lactate:    _lab(1.9),   // borderline
        pct:        _lab(0.84),  // elevated
        wbc:        _lab(14.6),  // leukocytosis
        platelets:  _lab(192),
        creatinine: _lab(1.3),   // mild AKI
        bilirubin:  _lab(0.8),
        crp:        _lab(68),    // elevated
        dengueNS1:  _noLab(0),
        malariaRDT: _noLab(0),
      },
    },

    // ── T+60min  Hemodynamic instability ────────────────────────────────────
    {
      vitals: {
        hr: 118, map: 68, resp: 25, temp: 39.1,
        o2sat: 95, urineOutput: 0.44, gcs: 15, systolicBp: 102,
      },
      labs: {
        lactate:    _lab(1.9),
        pct:        _lab(0.84),
        wbc:        _lab(14.6),
        platelets:  _lab(192),
        creatinine: _lab(1.3),
        bilirubin:  _lab(0.8),
        crp:        _lab(68),
        dengueNS1:  _noLab(0),
        malariaRDT: _noLab(0),
      },
    },

    // ── T+75min  SECOND LABS — sepsis criteria clearly met ──────────────────
    {
      vitals: {
        hr: 122, map: 65, resp: 26, temp: 39.3,
        o2sat: 94, urineOutput: 0.38, gcs: 14, systolicBp: 98,
      },
      labs: {
        lactate:    _lab(2.4),   // >2 confirms tissue hypoperfusion
        pct:        _lab(3.2),   // sepsis range
        wbc:        _lab(17.8),  // further rise
        platelets:  _lab(168),   // mild thrombocytopaenia
        creatinine: _lab(1.6),   // worsening AKI
        bilirubin:  _lab(1.0),
        crp:        _lab(148),   // markedly elevated
        dengueNS1:  _noLab(0),
        malariaRDT: _noLab(0),
      },
    },

    // ── T+90min  End-organ involvement — GCS dropping ───────────────────────
    {
      vitals: {
        hr: 126, map: 63, resp: 27, temp: 39.4,
        o2sat: 93, urineOutput: 0.32, gcs: 13, systolicBp: 95,
      },
      labs: {
        lactate:    _lab(2.4),
        pct:        _lab(3.2),
        wbc:        _lab(17.8),
        platelets:  _lab(168),
        creatinine: _lab(1.6),
        bilirubin:  _lab(1.0),
        crp:        _lab(148),
        dengueNS1:  _noLab(0),
        malariaRDT: _noLab(0),
      },
    },

    // ── T+105min  MAP < 65 — septic shock threshold ─────────────────────────
    {
      vitals: {
        hr: 131, map: 61, resp: 29, temp: 39.5,
        o2sat: 92, urineOutput: 0.26, gcs: 12, systolicBp: 92,
      },
      labs: {
        lactate:    _lab(2.4),
        pct:        _lab(3.2),
        wbc:        _lab(17.8),
        platelets:  _lab(168),
        creatinine: _lab(1.6),
        bilirubin:  _lab(1.0),
        crp:        _lab(148),
        dengueNS1:  _noLab(0),
        malariaRDT: _noLab(0),
      },
    },

    // ── T+120min  THIRD LABS — critical lactate, escalation required ─────────
    {
      vitals: {
        hr: 136, map: 58, resp: 31, temp: 39.6,
        o2sat: 91, urineOutput: 0.20, gcs: 12, systolicBp: 88,
      },
      labs: {
        lactate:    _lab(3.2),   // critical (>2 = shock)
        pct:        _lab(8.6),   // high sepsis burden
        wbc:        _lab(19.4),
        platelets:  _lab(142),   // thrombocytopaenia
        creatinine: _lab(2.1),   // significant AKI
        bilirubin:  _lab(1.4),   // mild liver involvement
        crp:        _lab(210),
        dengueNS1:  _noLab(0),
        malariaRDT: _noLab(0),
      },
    },

    // ── T+135min  Full septic shock ──────────────────────────────────────────
    {
      vitals: {
        hr: 140, map: 56, resp: 33, temp: 39.5,
        o2sat: 91, urineOutput: 0.16, gcs: 11, systolicBp: 86,
      },
      labs: {
        lactate:    _lab(3.2),
        pct:        _lab(8.6),
        wbc:        _lab(19.4),
        platelets:  _lab(142),
        creatinine: _lab(2.1),
        bilirubin:  _lab(1.4),
        crp:        _lab(210),
        dengueNS1:  _noLab(0),
        malariaRDT: _noLab(0),
      },
    },

    // ── T+150min  Post-intervention — partial stabilisation ──────────────────
    {
      vitals: {
        hr: 128, map: 64, resp: 28, temp: 39.1,
        o2sat: 93, urineOutput: 0.24, gcs: 12, systolicBp: 94,
      },
      labs: {
        lactate:    _lab(3.2),
        pct:        _lab(8.6),
        wbc:        _lab(19.4),
        platelets:  _lab(142),
        creatinine: _lab(2.1),
        bilirubin:  _lab(1.4),
        crp:        _lab(210),
        dengueNS1:  _noLab(0),
        malariaRDT: _noLab(0),
      },
    },

    // ── T+165min  Continued monitoring ───────────────────────────────────────
    {
      vitals: {
        hr: 120, map: 67, resp: 26, temp: 38.8,
        o2sat: 94, urineOutput: 0.30, gcs: 13, systolicBp: 98,
      },
      labs: {
        lactate:    _lab(3.2),
        pct:        _lab(8.6),
        wbc:        _lab(19.4),
        platelets:  _lab(142),
        creatinine: _lab(2.1),
        bilirubin:  _lab(1.4),
        crp:        _lab(210),
        dengueNS1:  _noLab(0),
        malariaRDT: _noLab(0),
      },
    },
  ],
};
