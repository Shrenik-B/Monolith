from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import init_db, get_db_connection

app = FastAPI()

# Enable CORS for Next.js
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

@app.get("/api/query-data")
def query_data():
    conn = get_db_connection()
    # Query DuckDB and convert results straight to a list of dicts/dataframes
    results = conn.execute("SELECT * FROM metrics").fetchall()
    conn.close()
    return {"data": results}