# Feature Store Specification (v1)

Each row in `features_v1.parquet` represents one business day.

All downstream modules (HMM, DTW, Scenario Simulator, Validation Dashboard) consume **only** this table. No model should directly query raw FRED or Yahoo data.

| Feature Category | Feature | Data Source | Raw Series | Frequency | Transformation Pipeline | Final Column |
|-----------------|----------|-------------|------------|-----------|-------------------------|--------------|
| **Inflation** | CPI | FRED | CPIAUCSL | Monthly | YoY % → 10Y Rolling Z-Score | `inflation_z` |
| | Core CPI | FRED | CPILFESL | Monthly | YoY % → 10Y Rolling Z-Score | `core_inflation_z` |
| **Growth** | Industrial Production | FRED | INDPRO | Monthly | YoY % → 10Y Rolling Z-Score | `industrial_production_z` |
| | Retail Sales | FRED | RSAFS | Monthly | YoY % → 10Y Rolling Z-Score | `retail_sales_z` |
| | PMI / ISM Manufacturing PMI | FRED / ISM | PMI | Monthly | 10Y Rolling Z-Score | `pmi_z` |
| **Labour Market** | Unemployment Rate | FRED | UNRATE | Monthly | 10Y Rolling Z-Score | `unemployment_z` |
| | Nonfarm Payrolls | FRED | PAYEMS | Monthly | YoY % → 10Y Rolling Z-Score | `payrolls_z` |
| **Monetary Policy** | Effective Fed Funds Rate | FRED | FEDFUNDS | Monthly | 10Y Rolling Z-Score | `fedfunds_z` |
| | Yield Curve | FRED | DGS10 − DGS2 | Daily | Spread → 10Y Rolling Z-Score | `yield_curve_z` |
| **Financial Conditions** | Credit Spread | FRED | BAA − AAA (or ICE BofA HY OAS) | Daily | Spread → 10Y Rolling Z-Score | `credit_spread_z` |
| | VIX | Yahoo Finance | ^VIX | Daily | 10Y Rolling Z-Score | `vix_z` |
| **Market Performance** | SPY | Yahoo Finance | SPY Close | Daily | 6M Return → 10Y Rolling Z-Score | `spy_return_z` |
| | Gold | Yahoo Finance | GLD (or GC Futures) | Daily | 6M Return → 10Y Rolling Z-Score | `gold_return_z` |
| | US Dollar Index | Yahoo Finance | DX-Y.NYB | Daily | 6M Return → 10Y Rolling Z-Score | `dxy_return_z` |
| | Crude Oil | Yahoo Finance | CL Futures (or USO) | Daily | 6M Return → 10Y Rolling Z-Score | `oil_return_z` |
| **Sentiment / Risk** | Economic Policy Uncertainty | EPU | US EPU Index | Monthly | 10Y Rolling Z-Score | `epu_z` |
| | Geopolitical Risk | GPR | GPR Index | Monthly | 10Y Rolling Z-Score | `gpr_z` |

---

## Metadata Columns

These are not model features but are stored alongside every row.

| Column | Type | Description |
|---------|------|-------------|
| `date` | DATE | Observation date (primary key) |
| `era_tag` | STRING | Structural macro era (Great Moderation, GFC, ZIRP, COVID, etc.) |
| `feature_version` | STRING | Feature schema version (e.g. `v1`) |
| `vintage_date` | DATE | ALFRED vintage used for macro features |
| `is_training_valid` | BOOLEAN | Indicates whether all required features are present for model training |

---

# Transformation Rules

## 1. YoY % Change

Used for level series that trend over time.

Formula:

YoY = (Current Value / Value 12 Months Ago) - 1

Applied to:

- CPI
- Core CPI
- Industrial Production
- Retail Sales
- Payrolls

---

## 2. Rolling Z-Score

Used to normalize features across different historical regimes.

Formula:

Z = (x - rolling_mean) / rolling_std

Rolling window:

- 10 years of available history

Applied after YoY where appropriate.

---

## 3. Spread Calculation

Used for variables whose relative difference is economically meaningful.

Examples:

Yield Curve

10Y Treasury − 2Y Treasury

Credit Spread

BAA Yield − AAA Yield

The spread is then normalized with a rolling z-score.

---

## 4. Rolling Return

Applied to market assets.

Formula:

6 Month Return = Price(t) / Price(t-6M) - 1

Then:

6 Month Return → Rolling Z-Score

Applied to:

- SPY
- Gold
- Dollar Index
- Oil

---

# Data Alignment Rules

- The feature table has **daily frequency**.
- Market data updates daily.
- Monthly macro data is updated **only on its official release date**.
- After release, macro values are forward-filled until the next release.
- Never backfill macro releases.
- Never interpolate macro data.

---

# Missing Data Policy

| Situation | Action |
|-----------|--------|
| Market holiday | No row (or handled by trading calendar) |
| Macro release not yet published | Forward-fill previous released value |
| Insufficient history for rolling calculations | Leave as `NULL` / `NaN` |
| API failure | Raise ingestion error; do not silently impute |

---

# Feature Store Contract

All downstream systems (HMM, DTW, Scenario Simulator, Daily Notes, Validation Dashboard) **must consume only `features_v1.parquet`**.

Raw source tables are used only by the ingestion and feature engineering pipeline.