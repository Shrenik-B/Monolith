import pandas as pd


def build_master_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return df.index


def align_to_index(
    df: pd.DataFrame,
    index: pd.DatetimeIndex,
    method: str = "ffill"
) -> pd.DataFrame:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas DatetimeIndex")
    return df.reindex(index, method=method)


def merge_features(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, axis=1)