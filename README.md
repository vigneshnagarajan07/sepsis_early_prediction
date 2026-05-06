# 🩺 Early Prediction of Sepsis

> **AI-powered clinical decision support system** for early sepsis detection, built for resource-limited settings.  
> Combines an XGBoost + LightGBM ensemble with qSOFA scoring, delivered through a real-time React dashboard.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements an **early-warning AI system** for sepsis prediction tailored to resource-limited clinical settings. By fusing routine vitals, bedside lab values, and basic demographics, the ensemble model produces a real-time sepsis probability score along with clinically interpretable feature attributions (surrogate SHAP drivers), alerting clinicians before the patient deteriorates.

---

## Features

- ⚡ **Ensemble ML Prediction** — XGBoost + LightGBM models with calibrated probability fusion
- 📊 **qSOFA Scoring** — Automated quick SOFA score calculation alongside AI score
- 🔥 **SHAP-like Feature Drivers** — Top influential clinical factors highlighted per patient
- 🌡️ **Alert Levels** — `none` / `warning` / `critical` triage flags
- 🧠 **Temporal Attention Heatmap** — 6-step attention weights visualisation
- 🌐 **React + Vite Dashboard** — Responsive, real-time UI with D3 visualisations
- 🔌 **FastAPI Backend** — Async REST API with full CORS support
- 🏥 **Tropical Disease Flags** — Dengue NS1 & Malaria RDT inputs for endemic regions

---

## Architecture

```
┌─────────────────────────┐        HTTP / JSON        ┌─────────────────────────┐
│   React Frontend (Vite) │ ────────────────────────▶ │  FastAPI Backend (Py)   │
│   Express Dev Proxy     │ ◀──────────────────────── │  XGBoost + LightGBM     │
│   D3 Visualisations     │      Prediction Result    │  Feature Engineering    │
└─────────────────────────┘                           └─────────────────────────┘
         :5173 / :3000                                          :8000
```

---

## Tech Stack

| Layer       | Technology                                         |
|-------------|----------------------------------------------------|
| Frontend    | React 19, TypeScript, Vite 6, Tailwind CSS 4, D3  |
| UI Motion   | Motion (Framer Motion), Lucide React               |
| Dev Proxy   | Express + http-proxy-middleware                    |
| Backend     | FastAPI, Uvicorn, Pydantic v2                      |
| ML Models   | XGBoost 2, LightGBM 4, scikit-learn, NumPy         |
| AI Features | Google Gemini API (`@google/genai`)                |

---

## Getting Started

### Prerequisites

- **Python** ≥ 3.10
- **Node.js** ≥ 18 & npm
- **Git**

---

### Backend Setup

```bash
# 1. Navigate to backend
cd backend

# 2. Create & activate virtual environment
python -m venv ../.venv

# Windows
..\.venv\Scripts\activate

# macOS / Linux
source ../.venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

### Frontend Setup

```bash
# 1. Navigate to frontend
cd frontend

# 2. Copy environment template and fill in your API key
cp .env.example .env

# 3. Install Node dependencies
npm install

# 4. Start the development server
npm run dev
```

The UI will be available at `http://localhost:3000` (proxies API calls to `:8000`).

---

## API Reference

### `GET /api/health`
Returns model readiness status.

```json
{ "status": "ok", "xgb_loaded": true, "lgbm_loaded": true, "models_ready": true }
```

### `POST /api/predict`
Submit a patient payload and receive a sepsis prediction.

**Request body:**
```json
{
  "vitals":       { "hr": 102, "map": 65, "resp": 22, "temp": 38.5, "o2sat": 94, "urineOutput": 0.4, "gcs": 14, "systolicBp": 90 },
  "labs":         { "lactate": {"value": 2.8, "performed": true}, "wbc": {"value": 14.5, "performed": true}, ... },
  "demographics": { "age": 55, "gender": "Male", "bmi": 26.0, "diabetes": true, ... }
}
```

**Response:**
```json
{
  "aiScore": 0.82,
  "qsofaScore": 2,
  "alertLevel": "critical",
  "attentionWeights": [...],
  "modelScores": { "xgb": 0.79, "lgbm": 0.85 },
  "featureSummary": { ... },
  "shapDrivers": [...]
}
```

### `GET /api/features`
Returns the feature name lists for both models (debugging / paper appendix).

---

## Project Structure

```
early-prediction-of-sepsis/
├── backend/
│   ├── main.py             # FastAPI app, routes, middleware
│   ├── models.py           # XGBoost & LightGBM loading + ensemble inference
│   ├── features.py         # Feature engineering & SHAP driver computation
│   ├── test_cases.py       # Unit tests for prediction pipeline
│   ├── requirements.txt    # Python dependencies
│   ├── xgb_booster.json    # Trained XGBoost model
│   └── lgbm_stream1.txt    # Trained LightGBM model
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Main React application
│   │   ├── main.tsx        # Entry point
│   │   └── index.css       # Global styles
│   ├── server.ts           # Express dev proxy server
│   ├── vite.config.ts      # Vite configuration
│   ├── package.json        # Node dependencies & scripts
│   └── .env.example        # Environment variable template
├── .gitignore
└── README.md
```

---

## Model Details

| Model      | Algorithm  | Features  | Notes                              |
|------------|------------|-----------|------------------------------------|
| XGBoost    | Gradient Boosted Trees | ~40 clinical features | Calibrated with Platt scaling |
| LightGBM   | Gradient Boosted Trees | ~40 clinical features | Leaf-wise growth, faster inference |
| **Ensemble** | Weighted average | — | Final score = 0.5×XGB + 0.5×LGBM |

Feature groups:
- **Vitals**: HR, MAP, Resp rate, Temperature, SpO₂, Urine output, GCS, Systolic BP
- **Labs**: Lactate, PCT, WBC, Platelets, Creatinine, Bilirubin, Dengue NS1, Malaria RDT
- **Demographics**: Age, Gender, BMI, Comorbidities (DM, CKD), Antibiotic history, Endemic flags

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "feat: add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ for improving critical care outcomes in resource-limited settings.</p>
