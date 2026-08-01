# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import init_db, get_db_connection

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database tables when FastAPI starts
@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/test")
def test_endpoint():
    return {"message": "Backend connected successfully!"}

@app.get("/api/duckdb-test")
def duckdb_test():
    conn = get_db_connection()
    # Query table and return formatted tuples
    results = conn.execute("SELECT * FROM metrics").fetchall()
    conn.close()
    return {"metrics": results}
