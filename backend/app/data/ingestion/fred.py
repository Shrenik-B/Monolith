from fredapi import Fred
import pandas as pd 
from pathlib import Path 
from datetime import timedelta 
from dotenv import load_dotenv
import os

load_dotenv()

fred = Fred(api_key=os.getenv("FRED_API_KEY"))


def download_fred_series(series_id : str, start_date : str, end_date : str):
    series = fred.get_series(series_id=series_id, observation_start=start_date, observation_end=end_date).to_frame(name = series_id)
    validate_series(series)
    return series 



def update_metric(metric : str, filepath : str):
    existing = load_fred_parquet(filepath)
    last_date = existing.index[-1]
    start = last_date.strftime("%Y-%m-%d")
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    new = download_fred_series(metric, start, end)
    if new.empty:
        return existing 
    updated = pd.concat([existing, new])
    updated = updated[~updated.index.duplicated(keep='last')]
    save_fred_parquet(updated, filepath)
    return updated

def save_fred_parquet(series : pd.Series, filepath : str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    series.to_parquet(filepath)
    return 

def load_fred_parquet(file : str):
    series = pd.read_parquet(file)
    return series

def validate_series(series : pd.Series):
    if series.empty:
        raise ValueError("Series is empty!")
    if series.index.has_duplicates:
        raise ValueError("Duplicate dates found!")
    if not series.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted!")
    return True 