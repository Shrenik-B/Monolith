# System Architecture

**Project:** Monolith

Version: v1

---

# High-Level Architecture

```

                    User

↓

Next.js Frontend

↓

FastAPI Backend

↓

Service Layer

↓

┌──────────────────────────────────────────────┐

│ Feature Store │ HMM │ DTW │ Scenario │ Search │

└──────────────────────────────────────────────┘

↓

DuckDB | PostgreSQL | Qdrant

↓

Raw Data Pipeline

↓

Yahoo Finance
FRED
ALFRED
EPU
GPR

```

---

# System Components

## 1. Data Ingestion

Responsible for downloading raw data.

Sources

- Yahoo Finance
- FRED
- ALFRED
- EPU
- GPR

Output

```
data/raw/
```

---

## 2. Feature Engineering

Converts raw data into normalized macro features.

Responsibilities

- Cleaning
- Alignment
- Forward filling
- Transformations
- Validation

Output

```
features_v1.parquet
```

---

## 3. Feature Store

Provides a single source of truth.

Responsibilities

- Load
- Save
- Validate
- Latest snapshot
- Historical windows

Consumed by

- HMM
- DTW
- Scenario Simulator

---

## 4. HMM Engine

Responsibilities

- Train
- Infer
- Predict regime probabilities
- Transition matrix

Consumes

```
features_v1.parquet
```

Produces

```
Current regime

Confidence

Transition probabilities
```

---

## 5. DTW Engine

Responsibilities

- Historical analogue search
- Similarity scoring

Consumes

```
features_v1.parquet
```

Produces

```
Ranked historical matches
```

---

## 6. Scenario Engine

Uses historical analogues to estimate

- Return distributions
- Historical outcomes

Consumes

- DTW
- HMM

---

## 7. RAG Engine

Responsibilities

- Semantic retrieval
- Literature search
- Evidence lookup

Uses

Qdrant

---

## 8. Daily Note Generator

Inputs

- HMM
- DTW
- Scenario
- RAG

Produces

Daily macro report

Stores

PostgreSQL

---

## 9. FastAPI

Acts as orchestration layer.

Responsibilities

- Validate requests
- Call services
- Return JSON

Contains **no business logic**.

---

## 10. Next.js Frontend

Responsibilities

Dashboard

Charts

Regime Timeline

Historical Analogues

Scenario Simulator

Daily Notes

Search

---

# Data Flow

```

Yahoo / FRED / ALFRED

↓

Raw Parquet

↓

Cleaning

↓

Alignment

↓

Transformations

↓

Validation

↓

features_v1.parquet

↓

┌──────────┬──────────┬─────────────┐

│ HMM │ DTW │ Scenario │

└──────────┴──────────┴─────────────┘

↓

Daily Note Generator

↓

FastAPI

↓

Next.js

↓

User

```

---

# Folder Responsibilities

```

backend/

app/

api/

services/

models/

core/

data/

quant/

rag/

tests/

```

---

# Dependency Rules

Allowed

```

API

↓

Services

↓

Feature Store

↓

DuckDB

```

Allowed

```

API

↓

Services

↓

Quant

```

Not Allowed

```

Frontend

↓

DuckDB

```

Not Allowed

```

Frontend

↓

HMM

```

Not Allowed

```

Quant

↓

Frontend

```

All communication goes through the service layer.

---

# Design Principles

1. Single source of truth for features.
2. Separation of concerns.
3. Stateless APIs.
4. Reproducible model training.
5. Point-in-time correctness.
6. Modular architecture.
7. Independent testing of every component.
8. Clear ownership between developers.

---

# Ownership

Developer 1

- Data ingestion
- Feature engineering
- DuckDB
- Feature Store

Developer 2

- HMM
- DTW
- Scenario Engine
- Validation

Developer 3

- FastAPI
- Next.js
- PostgreSQL
- Qdrant
- Deployment

---

# Future Extensions

- Multi-country macro regimes
- Portfolio optimization
- Reinforcement learning allocation
- Agentic research assistant
- Live streaming data
- User authentication
- Alerting and notifications