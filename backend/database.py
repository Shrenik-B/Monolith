# backend/database.py
import duckdb

DB_FILE = "monolith.duckdb"

def get_db_connection():
    """
    Connects to the DuckDB file on disk.
    If monolith.duckdb doesn't exist, DuckDB creates it automatically.
    """
    conn = duckdb.connect(DB_FILE)
    return conn

def init_db():
    """
    Initializes default tables if they do not exist yet.
    """
    conn = get_db_connection()
    # Example: Create a sample analytics/events table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY,
            event_name VARCHAR,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.close()