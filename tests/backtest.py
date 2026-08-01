"""
tests/backtest.py

Purpose:
--------
Walk-Forward Temporal Validation for the Gaussian HMM Market Regime Detection system.
"""

from __future__ import annotations

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
import numpy as np
import pandas as pd
from typing import Optional

from load import retrieve_features
from train import select_best_hmm
from predict import HMMPredictor, build_dynamic_regime_mapping, get_regime_name
import config

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

VALIDATION_SUITE = [
    ("2007-06-15", "Pre-GFC: Calm before the storm"),
    ("2008-09-15", "GFC: Lehman Collapse Day"),
    ("2009-03-09", "GFC: S&P Market Bottom"),
    ("2011-08-08", "US Credit Downgrade Sell-off"),
    ("2015-08-24", "China/Oil Flash Crash"),
    ("2020-03-16", "COVID Crash: Circuit Breaker Week"),
    ("2020-03-23", "COVID: Market Bottom"),
    ("2022-06-13", "Fed Hike Bear: Mid Drawdown"),
    ("2023-03-10", "SVB Banking Crisis"),
    ("2004-06-15", "Normal: Mid-2004 Quiet Expansion"),
    ("2005-09-15", "Normal: Mid-2005 Low Volatility Bull"),
    ("2006-07-14", "Normal: Pre-GFC Goldilocks Period"),
    ("2010-07-15", "Normal: Post-GFC Recovery (Low Vol)"),
    ("2012-03-15", "Normal: Euro Crisis Recovery / QE Bull"),
    ("2013-07-15", "Normal: Taper Tantrum Recovery"),
    ("2014-03-14", "Normal: Steady Expansion"),
    ("2016-07-15", "Normal: Post-Brexit Recovery Bull"),
    ("2017-07-14", "Normal: Low Volatility Bull Run Peak"),
    ("2019-06-14", "Normal: Pre-COVID Goldilocks"),
    ("2021-07-15", "Normal: Vaccine-era Bull Market"),
    ("2024-03-15", "Normal: Post-Inflation Normalization"),
    ("2025-06-13", "Normal: Current Expansion"),
]

def stress_category(name: str) -> str:
    n = name.lower()
    if "recessionary" in n or "bear" in n or "stress" in n or "stagflation" in n:
        return "BEAR/STRESS"
    elif "goldilocks" in n or ("low volatility" in n and "bull" in n) or "bull" in n:
        return "BULL/EXPANSION"
    elif "inflation" in n or "cyclical" in n or "late cycle" in n:
        return "INFLATIONARY"
    return "NEUTRAL"

def backtest_single_date(
    test_date_str: str, X_full: np.ndarray, df_full: pd.DataFrame, feature_names: list,
    full_model_predictor: Optional[HMMPredictor] = None, verbose: bool = True,
) -> dict:
    dates_str = df_full.index.strftime("%Y-%m-%d").tolist()
    if test_date_str in dates_str:
        test_idx = dates_str.index(test_date_str)
    else:
        target_dt = pd.to_datetime(test_date_str)
        test_idx = int(np.argmin(np.abs(df_full.index - target_dt)))
        test_date_str = df_full.index[test_idx].strftime("%Y-%m-%d")

    actual_test_date = df_full.index[test_idx].strftime("%Y-%m-%d")
    X_train = X_full[:test_idx]
    df_train = df_full.iloc[:test_idx]
    X_test_obs = X_full[test_idx: test_idx + 1]

    train_samples = len(X_train)
    if train_samples < 100:
        return {"date": actual_test_date, "train_samples": train_samples, "test_idx": test_idx, "error": "Insufficient training data"}

    try:
        wf_model, wf_metrics, _ = select_best_hmm(
            X=X_train, candidate_states=config.CANDIDATE_STATES,
            covariance_type=config.COVARIANCE_TYPE, random_state=config.RANDOM_STATE,
        )
    except Exception as e:
        return {"date": actual_test_date, "train_samples": train_samples, "test_idx": test_idx, "error": str(e)}

    wf_mapping = build_dynamic_regime_mapping(wf_model, X_train, feature_names, df_train)
    X_eval = np.vstack([X_train, X_test_obs])
    wf_state_seq_eval = wf_model.predict(X_eval)
    wf_posterior = wf_model.predict_proba(X_eval)

    wf_state = int(wf_state_seq_eval[-1])
    wf_confidence = float(np.max(wf_posterior[-1]))
    wf_regime_name = get_regime_name(wf_state, wf_mapping)

    return {
        "date": actual_test_date, "train_samples": train_samples,
        "wf_regime": wf_regime_name, "wf_confidence": wf_confidence,
    }

if __name__ == "__main__":
    X_full, feature_names, df_full = retrieve_features(config.PARQUET_FILE, use_only_z_features=config.USE_ONLY_Z_FEATURES)
    print(f"Loaded {len(X_full):,} rows for backtesting.")
    res = backtest_single_date("2020-03-16", X_full, df_full, feature_names)
    print("Backtest Result for 2020-03-16:", res)
