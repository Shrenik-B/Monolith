# Implementation Plan - Convert Streamlit Dashboard to React (Next.js) & FastAPI

Convert the existing Streamlit dashboard (`app.py`) into a full-stack **React (Next.js 16)** frontend coupled with a **FastAPI** backend. The system will retain 100% feature parity with the quantitative pipeline (K-Means++ Init, Baum-Welch Gaussian HMM, BIC Model Selection, Past-Only DTW Trajectory Engine, and AI Analyst Executive Commentary) while delivering a responsive web design with dark-mode glassmorphism aesthetics.

## User Review Required

> [!IMPORTANT]
> - **Backend Architecture**: FastAPI will load the dataset (`features_v1.parquet`) and HMM model (`model.pkl` / `model_metadata.json`) into memory on startup to provide sub-millisecond API responses for date selection, time-series charts, DTW trajectory matching, and transition matrix queries.
> - **Frontend Stack**: Next.js 16 + React 19 + Tailwind CSS + Recharts (for financial & statistical charts) + Lucide React (for icons).
> - **Streamlit App**: The original `app.py` remains in the workspace as reference, but the primary user interface will now be served via React at `http://localhost:3000` connected to FastAPI at `http://localhost:8000`.

---

## Architecture & API Specification

### FastAPI Backend Endpoints (`backend/main.py`)
1. `GET /api/health` — Health check endpoint.
2. `GET /api/dates` — Returns available trading date range (`min_date`, `max_date`) and all date strings.
3. `GET /api/metadata` — Returns HMM model parameters, AIC/BIC metrics, convergence status, and feature names.
4. `GET /api/history` — Returns full dataset time-series for cumulative market index, 30-day rolling return, regime classifications, indicator features, and historical regime frequency counts.
5. `GET /api/inference?date=YYYY-MM-DD` — Returns single-date market regime intelligence:
   - Current regime & posterior confidence score
   - Realized next-day regime outcome (for historical dates) or 1-day forward Markov forecast
   - Full state probability distribution
   - Current macro feature vector & normalized risk radar values
6. `GET /api/dtw?date=YYYY-MM-DD&window_size=30&top_k=3` — Returns strict past-only DTW trajectory matches and aligned VIX time-series arrays for plotting.
7. `GET /api/transition-matrix` — Returns state transition probability matrix, regime labels, and daily persistence metrics.
8. `POST /api/ai-report` (or `GET /api/ai-report?date=YYYY-MM-DD`) — Generates automated AI market analyst executive commentary (overview, outlook, analogs, feature drivers).

---

## Proposed Changes

### Backend Component (`backend`)

#### [MODIFY] [main.py](file:///c:/Users/bshre/Downloads/Monolith/backend/main.py)
- Wire up data loading (`retrieve_features`), `HMMPredictor`, `DTWEngine`, and `generate_ai_market_commentary`.
- Implement REST API endpoints listed above with proper CORS middleware for Next.js.

---

### Frontend Component (`frontend`)

#### [MODIFY] [package.json](file:///c:/Users/bshre/Downloads/Monolith/frontend/package.json)
- Add `recharts` for charting and `lucide-react` for UI icons.

#### [MODIFY] [globals.css](file:///c:/Users/bshre/Downloads/Monolith/frontend/app/globals.css)
- Implement dark theme palette (navy/slate/emerald/crimson accents), custom glassmorphism card styles, custom scrollbars, and badge styling.

#### [MODIFY] [layout.tsx](file:///c:/Users/bshre/Downloads/Monolith/frontend/app/layout.tsx)
- Set page title, metadata, fonts, and dark background container.

#### [NEW] [page.tsx](file:///c:/Users/bshre/Downloads/Monolith/frontend/app/page.tsx)
- Main Dashboard component featuring:
  - **Header & Calendar Date Selector**: Select target market date from 2003 to 2026 with instant reactivity.
  - **Model Summary & Regime Legend Sidebar**: Model parameters (States, Samples, AIC, BIC, Log-Likelihood) and regime badges.
  - **Top Intelligence Metric Cards**: Current Regime, Confidence Score, Realized Shift / Tomorrow's Forecast, VIX Percentile, Credit Spread.
  - **Educational Guide Modal / Collapsible**: Details on economic regimes and stress indicators.
  - **5 Tabbed Views**:
    1. **📊 Market & Regime Charts**: Cumulative growth chart with regime color-coding, 30-day rolling return area chart, interactive macro indicator selector, regime probability distribution, historical regime pie/donut chart.
    2. **🔮 Tomorrow's Forecast**: Realized ground-truth historical outcome banner, Markov projection card, tomorrow's probability distribution, transition matrix heatmap grid, daily state persistence table.
    3. **🔍 DTW Trajectory Search**: Window size & top-k sliders, VIX trajectory alignment chart (query vs historical matches), historical match cards with forward 30-day SPY returns.
    4. **🤖 AI Market Analyst**: Interactive "Generate Commentary" button, Executive overview, Tomorrow's outlook, Historical precedent, Macro drivers, Macro risk radar (RadarChart).
    5. **⚙️ Macro Drivers & Model**: Standardized feature vector bar chart, dataset statistics, dynamic regime mappings list, raw feature & model metadata JSON inspection.

---

## Verification Plan

### Automated Tests
- Test FastAPI endpoints via Python script (`pytest` or standalone `httpx` script) to verify status codes and JSON schema structure for `/api/dates`, `/api/inference`, `/api/dtw`, `/api/history`, `/api/transition-matrix`, and `/api/ai-report`.

### Manual Verification
- Launch FastAPI backend: `python -m uvicorn backend.main:app --port 8000 --reload`
- Launch Next.js frontend: `npm run dev` in `frontend` directory.
- Verify date picking updates all cards, charts, DTW trajectories, and AI reports seamlessly.
- Verify responsive layout, tab navigation, hover tooltips, and color-coded regime badges.
