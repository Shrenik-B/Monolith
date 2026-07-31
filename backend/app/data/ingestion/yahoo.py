import yfinance as yf 
import pandas as pd 
from datetime import timedelta 
from pathlib import Path 

def download_asset(ticker : str, start_date : str, end_date : str):
    df = yf.download(ticker, start = start_date, end = end_date, auto_adjust=True, back_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[["Close"]]
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    validate_dataframe(df)
    return df

def download_assets(tickers : list[str], start_date : str, end_date : str):
    assets = {}
    for ticker in tickers:
        assets[ticker] = download_asset(ticker, start_date, end_date)
    return assets   

def align_assets(assets : dict):
    combined_data = pd.concat(objs = assets, axis = 1)
    return combined_data 

def validate_dataframe(df : pd.DataFrame):
    if df.empty:
        raise ValueError("DataFrame is empty!")
    if df.index.has_duplicates:
        raise ValueError("Duplicate dates found!")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted!")
    required_columns = {'Close'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing columns : {required_columns - set(df.columns)}")
    if df[list(required_columns)].isnull().any().any():
        raise ValueError("Missing Values detected")

    return True 

def update_asset(ticker : str, filepath : str):
    existing = load_parquet(filepath)
    last_date = existing.index[-1]
    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    new = download_asset(ticker, start, end)
    if new.empty:
        return existing 
    updated = pd.concat([existing, new])
    updated = updated[~updated.index.duplicated(keep='last')]
    save_parquet(updated, filepath)
    return updated


def get_asset(ticker, start_date, end_date):

    filepath = Path(f"data/raw/yahoo/{ticker}.parquet")

    if filepath.exists():
        print("Loading from cache...")
        return load_parquet(filepath)

    print("Downloading...")
    df = download_asset(ticker, start_date, end_date)
    save_parquet(df, filepath)

    return df    


def save_parquet(df : pd.DataFrame, filepath : str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(filepath)
    return 

def load_parquet(file : str):
    df = pd.read_parquet(file)
    return df