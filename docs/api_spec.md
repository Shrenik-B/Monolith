# API Specification (v1)

**Project:** Monolith

**Version:** v1

---

# Overview

The FastAPI backend exposes REST endpoints that provide access to:

- Market & macroeconomic data
- Regime classification
- Historical analogue search
- Scenario simulation
- Daily macro research notes
- Literature search
- Validation results

All responses use JSON.

Base URL:

```
/api/v1
```

---

# Endpoint Groups

```
/health
/data
/regime
/analogue
/scenario
/daily-note
/search
/validation
```

---

# 1. Health

## GET /health

Returns backend status.

### Response

```json
{
    "status": "healthy",
    "version": "1.0.0"
}
```

---

# 2. Data Endpoints

## GET /data/latest

Returns the latest feature vector.

### Response

```json
{
    "date": "2026-07-29",
    "features": {
        "inflation_z": 1.42,
        "core_inflation_z": 1.31,
        "fedfunds_z": 0.74,
        "...": "..."
    }
}
```

---

## GET /data/window

Returns feature vectors over a specified date range.

### Query Parameters

```
start_date
end_date
```

Example

```
GET /data/window?start_date=2020-01-01&end_date=2021-01-01
```

### Response

```json
[
    {
        "date":"2020-01-02",
        "inflation_z":0.41,
        "fedfunds_z":-0.55
    },
    ...
]
```

---

# 3. Regime Endpoints

## POST /regime/classify

Assigns a regime to a specified date.

### Request

```json
{
    "date":"2026-07-29"
}
```

### Response

```json
{
    "date":"2026-07-29",
    "regime":"Inflationary Tightening",
    "regime_id":3,
    "confidence":0.91
}
```

---

## GET /regime/current

Returns the latest regime.

### Response

```json
{
    "date":"2026-07-29",
    "regime":"Inflationary Tightening",
    "confidence":0.91,
    "changed":false
}
```

---

## GET /regime/transition

Returns HMM transition probabilities.

### Response

```json
{
    "current_regime":"Inflationary Tightening",

    "one_month":{

        "Inflationary Tightening":0.83,

        "Disinflation":0.12,

        "Recession":0.05

    },

    "expected_duration_months":8.4
}
```

---

# 4. Historical Analogue

## POST /analogue/search

Finds historical periods similar to the current macro environment.

### Request

```json
{
    "window_months":6,

    "top_n":10,

    "features":[

        "inflation_z",

        "fedfunds_z",

        "vix_z"

    ]
}
```

### Response

```json
{
    "matches":[

        {

            "date":"2006-09-01",

            "similarity":94.8,

            "era":"Great Moderation",

            "forward_returns":{

                "SPY":0.12,

                "Gold":0.07

            }

        }

    ]
}
```

---

# 5. Scenario Simulator

## POST /scenario/run

Runs a historical scenario query.

### Request

```json
{
    "shock":"Fed Funds",

    "change":0.50,

    "window_months":3,

    "condition_on_regime":true
}
```

### Response

```json
{
    "sample_size":18,

    "distribution":{

        "worst":-0.22,

        "median":0.04,

        "best":0.19

    }
}
```

---

# 6. Daily Research Note

## GET /daily-note/latest

Returns the most recent generated note.

### Response

```json
{
    "date":"2026-07-29",

    "regime":"Inflationary Tightening",

    "changed":false,

    "summary":"...",

    "drivers":[

        "...",

        "..."

    ],

    "citations":[

        "...",

        "..."

    ]
}
```

---

## GET /daily-note/history

Returns previous notes.

Query Parameters

```
limit
offset
```

---

# 7. Literature Search

## POST /search

Semantic search over embedded macroeconomic literature.

### Request

```json
{
    "query":"inflation persistence"
}
```

### Response

```json
{
    "results":[

        {

            "source":"FOMC",

            "date":"2022-11",

            "page":18,

            "text":"..."

        }

    ]
}
```

---

# 8. Validation Dashboard

## GET /validation/latest

Returns the latest walk-forward validation results.

### Response

```json
{
    "reaction_lag":1.8,

    "whipsaw_rate":0.12,

    "brier_score":0.21,

    "baseline_brier_score":0.34
}
```

---

## GET /validation/timeline

Returns regime assignments over time.

### Response

```json
[
    {

        "date":"2008-09-15",

        "regime":"Financial Crisis"

    },

    ...

]
```

---

# Common Response Codes

| Code | Meaning |
|-------|----------|
| 200 | Success |
| 201 | Created |
| 400 | Invalid request |
| 404 | Resource not found |
| 422 | Validation error |
| 500 | Internal server error |

---

# Authentication

v1:

No authentication required.

Future versions:

- JWT
- OAuth
- API Keys

---

# Versioning

All endpoints are versioned.

```
/api/v1
```

Future incompatible changes become

```
/api/v2
```

---

# Design Principles

- The frontend communicates **only** with FastAPI.
- FastAPI communicates with:
    - Feature Store
    - HMM Engine
    - DTW Engine
    - Scenario Engine
    - Qdrant
    - PostgreSQL
- Raw FRED/Yahoo data is never exposed directly.
- All model outputs are derived from `features_v1.parquet`.
- All responses are JSON.
- Endpoints should remain deterministic and stateless where possible.