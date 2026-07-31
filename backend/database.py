# backend/database.py
import duckdb

DB_FILE = "monolith.duckdb"

def get_db_connection():
    """
    Returns a connection to the monolith.duckdb file on disk.
    Creates the file automatically if it doesn't exist.
    """
    return duckdb.connect(DB_FILE)

def init_db():
    """
    Initializes default tables at startup.
    """
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY,
            event_name VARCHAR,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.close()