"""
Root compatibility bridge for pipeline.py.
Executes run_pipeline from backend.app.pipeline.
"""

from backend.app.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
