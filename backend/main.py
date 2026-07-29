from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Next.js regardless of which IP or port it loads on
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from datetime import datetime

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/test")
def get_test_data():
    return {
        "message": "Hello from FastAPI Backend!",
        "variable_value": 40,
        "server_time": datetime.now().strftime("%H:%M:%S"),
        "status": "connected",
        "environment": "localhost"
    }