import pandas as pd
from pathlib import Path

# Replace these with the official download URLs
MONTHLY_GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
DAILY_GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


def download_gpr_monthly() -> pd.DataFrame:
    """
    Download the latest monthly GPR dataset.
    """
    df = pd.read_excel(MONTHLY_GPR_URL)
    df = df[df["month"] >= "1985-01-01"]
    df = df.set_index("month").sort_index()
    df = df[["GPR"]]
    validate_gpr_dataframe(df)

    return df


def download_gpr_daily() -> pd.DataFrame:
    """
    Download the latest daily GPR dataset.
    """
    df = pd.read_excel(DAILY_GPR_URL)
    df["DAY"] = pd.to_datetime(
                df["DAY"].astype(str),
                format="%Y%m%d"
            )
    df = df.set_index('DAY').sort_index()
    df = df[["GPRD"]]
    validate_gpr_dataframe(df)

    return df


def update_gpr_monthly(filepath: str) -> pd.DataFrame:
    """
    Downloads the latest monthly dataset and overwrites the cache.
    """
    df = download_gpr_monthly()

    save_gpr_parquet(df, filepath)

    return df


def update_gpr_daily(filepath: str) -> pd.DataFrame:
    """
    Downloads the latest daily dataset and overwrites the cache.
    """
    df = download_gpr_daily()

    save_gpr_parquet(df, filepath)

    return df


def save_gpr_parquet(df: pd.DataFrame, filepath: str):

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(filepath)

    return


def load_gpr_parquet(filepath: str) -> pd.DataFrame:

    return pd.read_parquet(filepath)


def validate_gpr_dataframe(df: pd.DataFrame):

    if df.empty:
        raise ValueError("Downloaded dataframe is empty.")

    if df.index.has_duplicates:
        raise ValueError("Duplicate dates found.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted.")

    return True