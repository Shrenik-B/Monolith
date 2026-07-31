import pandas as pd 
import numpy as np 

def pct_change(df : pd.DataFrame, periods : int = 1) -> pd.DataFrame:
    return df.pct_change(
                periods=periods,
                fill_method=None
            )

def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df / df.shift(1))

def yoy_change(df: pd.DataFrame, periods : int = 12) -> pd.DataFrame:
    return df.pct_change(periods = periods)

def rolling_zscore(df : pd.DataFrame, window : int) -> pd.DataFrame:
    mean = df.rolling(window).mean()
    std = df.rolling(window).std()
    return (df-mean)/std

def rolling_percentile(df : pd.DataFrame, window : int) -> pd.DataFrame:
    return df.rolling(window).rank(pct = True)

def moving_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window).mean()

def yield_curve(long_rate : pd.DataFrame, short_rate : pd.DataFrame) -> pd.DataFrame:
    return long_rate - short_rate

def credit_spread(corporate: pd.DataFrame, treasury: pd.DataFrame) -> pd.DataFrame:
    return corporate - treasury

def rolling_volatility(returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    return returns.rolling(window).std() * np.sqrt(252)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()

TRANSFORMS = {
    "pct_change": pct_change,
    "log_returns": log_returns,
    "yoy_change": yoy_change,
    "rolling_zscore": rolling_zscore,
    "rolling_percentile": rolling_percentile,
    "moving_average": moving_average,
    "yield_curve": yield_curve,
    "credit_spread": credit_spread,
    "rolling_volatility": rolling_volatility,
    "normalize": normalize,
}