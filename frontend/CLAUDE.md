# Backend Dependencies
- FastAPI / Uvicorn
- DuckDB (`monolith.duckdb` created dynamically on server start)

# Running Backend
`cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000`