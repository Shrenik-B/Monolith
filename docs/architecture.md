# Database Schema (v1)

**Project:** Monolith

---

# Overview

The system uses three storage layers:

| Storage | Purpose |
|----------|---------|
| DuckDB + Parquet | Historical market & macro feature data |
| PostgreSQL | Application metadata, reports, notes |
| Qdrant | Vector embeddings for semantic search |

Raw market and macroeconomic time series are **not** stored in PostgreSQL.

---

# PostgreSQL Tables

## daily_notes

Stores generated daily macro reports.

| Column | Type | Description |
|---------|------|-------------|
| id | UUID | Primary Key |
| date | DATE | Report date |
| regime | VARCHAR | Assigned regime |
| confidence | FLOAT | HMM confidence |
| changed | BOOLEAN | Whether regime changed |
| summary | TEXT | Executive summary |
| created_at | TIMESTAMP | Creation timestamp |

---

## daily_note_sources

Stores citations used in each report.

| Column | Type |
|---------|------|
| id | UUID |
| note_id | UUID (FK) |
| source | TEXT |
| page | INTEGER |
| quote | TEXT |

---

## regime_history

Stores every historical regime assignment.

| Column | Type |
|---------|------|
| date | DATE (PK) |
| regime_id | INTEGER |
| regime_name | VARCHAR |
| confidence | FLOAT |

---

## scenario_runs

Stores scenario simulator executions.

| Column | Type |
|---------|------|
| id | UUID |
| created_at | TIMESTAMP |
| scenario | JSONB |
| results | JSONB |

---

## analogue_searches

Stores historical analogue searches.

| Column | Type |
|---------|------|
| id | UUID |
| created_at | TIMESTAMP |
| parameters | JSONB |
| matches | JSONB |

---

## model_metadata

Stores trained model information.

| Column | Type |
|---------|------|
| model_name | VARCHAR |
| version | VARCHAR |
| trained_until | DATE |
| created_at | TIMESTAMP |
| parameters | JSONB |

---

## feature_versions

Tracks feature store versions.

| Column | Type |
|---------|------|
| version | VARCHAR |
| created_at | TIMESTAMP |
| description | TEXT |

---

# DuckDB

DuckDB is the analytical database.

No application metadata is stored here.

---

## Raw Tables

Example

```
spy_prices

date

open

high

low

close

volume
```

Each market series has its own table or Parquet file.

---

## Processed Tables

Primary dataset

```
features_v1
```

Contains

```
date

inflation_z

core_inflation_z

...

gpr_z
```

This is the only dataset consumed by the quant models.

---

# Qdrant Collections

## macro_documents

Stores embeddings for

- FOMC Minutes
- IMF Reports
- ECB Publications
- BIS Papers
- Federal Reserve Speeches
- Research Papers

Payload

```json
{
    "title":"",
    "date":"",
    "source":"",
    "page":12,
    "chunk":"..."
}
```

---

## daily_note_embeddings

Optional future collection.

Stores embeddings of generated daily notes.

---

# Relationships

daily_notes

↓

daily_note_sources

---

regime_history

↓

daily_notes

---

model_metadata

↓

regime_history

---

feature_versions

↓

features_v1.parquet

---

# Data Ownership

DuckDB

- Raw market data
- Raw macro data
- Feature store

PostgreSQL

- Metadata
- Reports
- Regime history
- Scenarios

Qdrant

- Vector search
- Literature
- Semantic retrieval