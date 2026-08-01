# 📈 Monolith — Version 1 (Production MVP)

A production-grade **macroeconomic market regime detection, transition forecasting, and historical similarity dashboard** powered by a Gaussian Hidden Markov Model (HMM), K-Means++ initialization, Dynamic Time Warping (DTW) trajectory search, and an AI Market Analyst synthesis engine.

---

## 📌 Executive Summary & Core Capabilities

This system processes **5,525 trading days (2003–2026)** across **17 macroeconomic and financial indicators** to detect statistical economic regimes, project 1-day forward Markov transitions, track historical ground-truth regime outcomes, and match trajectory analogs using Dynamic Time Warping (DTW).

### Key Features:
- **7-State Gaussian HMM**: Detects Bull, Goldilocks, Inflationary Expansion, Late Cycle, and Recessionary Bear regimes.
- **K-Means++ Deterministic Seeding**: Eliminates random initialization traps and ensures robust Baum–Welch EM convergence.
- **Automated BIC Model Selection**: Grid sweeps $K \in \{3, 4, 5, 6, 7\}$ hidden states to pick the optimal parsimonious fit ($K=7$, BIC=`192,074.58`).
- **Dynamic Empirical Auto-Mapping**: Maps HMM state permutations to human-readable economic regimes based on empirical stress scores.
- **Ground-Truth Realized Next-Day Transition Tracking**: Identifies realized state shifts ($T \to T+1$) when inspecting past market dates.
- **Strict Past-Only DTW Trajectory Matching**: Matches historical trajectory windows that ended *before* the query start date, reporting post-window 30-day SPY returns.
- **Interactive Streamlit Intelligence Dashboard**: 5 tabbed views with Plotly charts, radar risk scores, transition heatmaps, and on-demand AI executive commentary.

---

## 📂 Project Directory Structure

```text
├── config.py             # Global Configuration & Model Parameters
├── load.py               # Feature Dataset Extraction & Normalization
├── init.py               # K-Means++ Seeding Initializer
├── train.py              # Baum-Welch HMM Training & Model Selection (AIC/BIC)
├── model_persistence.py  # Binary & JSON Persistence (model.pkl / model_metadata.json)
├── predict.py            # HMM Inference Engine, Realized Transitions & Caching
├── dtw_engine.py         # Dynamic Time Warping Trajectory Matching (Past-Only)
├── ai_analyst.py         # AI Executive Macro Report Synthesis Engine
├── pipeline.py           # End-to-End Execution Pipeline Orchestrator
├── app.py                # Streamlit Web Intelligence Dashboard
├── features_v1.parquet   # Macro Dataset (5,525 rows x 17 features, 2003-2026)
├── model.pkl             # Trained Gaussian HMM Binary
├── model_metadata.json   # Model Metadata, Metrics, & Full Parameters JSON
├── README.md             # Project Documentation
└── tests/                # Testing & Validation Suite
    ├── evaluate.py       # Formal Train/Validate/Test Temporal Evaluation
    ├── backtest.py       # Walk-Forward Single/Multi-Date Backtest Suite
    └── kmean.py          # Standalone K-Means++ Seeding Test
```

---

## 🛠️ Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
```bash
python pipeline.py
```

### 3. Launch the Streamlit Intelligence Dashboard
```bash
streamlit run app.py
```

### 4. Run the Evaluation & Testing Suite
To execute temporal Train/Val/Test validation, walk-forward testing, or K-Means++ initialization tests:
```bash
python tests/evaluate.py
python tests/backtest.py
python tests/kmean.py
```

---

## 🏗️ System Architecture & Workflow

```text
               features_v1.parquet (5,525 × 17 matrix)
                                 │
                                 ▼
                    load.py (Standardization)
                                 │
                                 ▼
                   init.py (K-Means++ Centroids)
                                 │
                                 ▼
             train.py (Baum-Welch EM, BIC Selection)
                                 │
                                 ▼
            model_persistence.py (model.pkl & metadata.json)
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
 predict.py (Viterbi + Posterior)          dtw_engine.py (Trajectory Match)
            │                                         │
            └────────────────────┬────────────────────┘
                                 ▼
                  ai_analyst.py (Macro Synthesis)
                                 │
                                 ▼
                   app.py (Streamlit Dashboard)
```

---

## 📊 Dataset & 17 Macro Features

| Feature | Description | Macro Sign / Role |
|---|---|---|
| `spy_returns` | Daily S&P 500 equity index return | Equity Momentum |
| `gold_returns` | Daily Gold commodity return | Safe-Haven / Inflation |
| `dxy_returns` | US Dollar Index return | Currency Strength |
| `oil_returns` | Crude Oil commodity return | Energy Inflation |
| `vix_percentile` | VIX Volatility percentile (2003–2026) | Market Fear / Hedging Demand |
| `inflation_z` | Headline CPI Inflation z-score | Price Pressure |
| `core_inflation_z` | Core CPI Inflation z-score | Underlying Price Trend |
| `industrial_production_z` | Industrial output z-score | Real Economic Output |
| `retail_sales_z` | Consumer retail sales z-score | Consumer Health |
| `unemployment_z` | Unemployment rate z-score | Labor Market Slack |
| `payrolls_z` | Non-farm payroll z-score | Employment Growth |
| `housing_z` | Housing starts & permits z-score | Credit & Real Estate |
| `fedfunds_z` | Federal Funds Rate z-score | Monetary Policy Stance |
| `gpr_monthly_z` | Geopolitical Risk Index (Monthly) | Macro Risk Premium |
| `gpr_daily_z` | Geopolitical Risk Index (Daily) | Intraday Risk Shock |
| `yield_curve` | 10Y–2Y Treasury yield spread | Inversion = Recession Signal |
| `credit_spread` | High-Yield Corporate Bond spread (%) | Credit Stress & Liquidity |

---

## 🔬 Model Performance & Evaluation Results

### Model Selection ($K \in \{3, 4, 5, 6, 7\}$)
Grid search selection over 5,525 trading days selects **$K = 7$ states** based on minimum BIC:

| Hidden States ($K$) | Free Parameters | Log-Likelihood | AIC Score | BIC Score | Selection Rank |
|---|---|---|---|---|---|
| **$K = 7$ (Selected)** | **1,238** | **`-90,703.34`** | **`183,882.69`** | **`192,074.58`** | **#1 Optimal** |
| **$K = 6$** | 1,055 | `-93,422.10` | `188,954.20` | `195,935.18` | #2 |
| **$K = 4$** | 695 | `-96,298.20` | `193,986.40` | `198,585.25` | #3 |
| **$K = 5$** | 874 | `-95,908.93` | `193,565.87` | `199,349.16` | #4 |
| **$K = 3$** | 518 | `-99,622.80` | `200,281.59` | `203,709.22` | #5 |

### Chronological Train / Validate / Test Performance

| Split | Horizon | Observations ($N$) | Log-Likelihood / Obs | Category Precision | Daily Stability |
|---|---|---|---|---|---|
| **TRAIN (70%)** | `2003-01-29` ➔ `2018-07-18` | 3,867 | `-14.65` | **52.3%** | **90.8%** |
| **VALIDATE (15%)** | `2018-07-19` ➔ `2021-11-09` | 828 | `-30.64` | **37.2%** | **90.4%** |
| **TEST (15%)** | `2021-11-10` ➔ `2026-07-31` | 830 | `-29.03` | **44.9%** | **90.0%** |

- **Regime Stability**: Average **`90.4%`** consecutive non-flicker days, ensuring non-noisy macro signals.
- **Bear / Crisis Precision**: **`73.3% - 91.7%`** precision in flagging recessionary stress periods.

---

## 🎛️ Dashboard Overview (`app.py`)

The Streamlit web application includes 5 specialized interactive tabs:

1. **📊 Regime & Market Charts**: Cumulative S&P 500 growth colored by regime, 30-day rolling returns, interactive macro feature scatter/lines, state probability bars, and regime frequency pie chart.
2. **🔮 Tomorrow's Forecast**: Realized historical next-day regime outcome banners ($T+1$), 1-day Markov forward projection, state persistence metrics, and transition probability heatmap.
3. **🔍 DTW Trajectory Search**: VIX trajectory alignment chart (query vs historical matches), Top-K match cards, DTW distances, and post-window 30-day SPY return outcomes.
4. **🤖 AI Market Analyst**: On-demand executive commentary, crisis alerts, macro driver deep-dive, historical precedent analysis, and radar risk chart.
5. **⚙️ Macro Drivers & Model**: Full standardized feature vector bars, model parameters, dynamic regime mapping dictionary, and raw JSON metadata viewer.

---

## 🚀 Setup & Execution Guide

### 1. Installation
```bash
pip install streamlit hmmlearn scikit-learn pandas numpy plotly fastdtw scipy
```

### 2. Run the Full End-to-End Pipeline
Train the Gaussian HMM, perform model selection, save `model.pkl` & `model_metadata.json`, and run DTW matching:
```bash
python pipeline.py
```

### 3. Launch the Streamlit Intelligence Dashboard
```bash
streamlit run app.py
```

### 4. Run the Evaluation & Testing Suite
To execute temporal Train/Val/Test validation, walk-forward testing, or K-Means++ initialization tests:
```bash
python tests/evaluate.py
python tests/backtest.py
python tests/kmean.py
```
=======
# Monolith
--
>>>>>>> origin/model
