from __future__ import annotations

"""
audit_and_validate.py

Comprehensive Audit & Validation Suite for Gaussian HMM Market Regime Detection:
  1. Historical Crisis Date Alignment & Sanity Check
  2. Simple Threshold Rule Baseline Comparison & Confusion Matrix
  3. Feature Importance / Means Breakdown per Regime
  4. Multi-Seed Regime Label Stability Test
"""

import os
import sys
import io
from pathlib import Path

# Bootstrap project root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from backend.app.quant.regime.load import retrieve_features
from backend.app.quant.regime.predict import HMMPredictor, get_regime_name, build_dynamic_regime_mapping
from backend.app.quant.regime.train import train_single_hmm
from backend.app.core import config

HISTORICAL_CRISES = [
    {"name": "2008 Global Financial Crisis (GFC)", "start": "2007-12-01", "end": "2009-06-30"},
    {"name": "2011 US Credit Downgrade / Euro Debt", "start": "2011-07-01", "end": "2011-10-31"},
    {"name": "2015-16 China Devaluation / Oil Collapse", "start": "2015-08-01", "end": "2016-02-28"},
    {"name": "2020 COVID-19 Liquidity Shock", "start": "2020-02-15", "end": "2020-04-30"},
    {"name": "2022 Fed Rate-Hike Inflation Bear Market", "start": "2022-01-01", "end": "2022-11-30"},
    {"name": "2023 SVB Regional Banking Shock", "start": "2023-03-01", "end": "2023-05-15"},
]


def audit_crisis_alignment(df_regimes: pd.DataFrame):
    """Audit #1: Checks HMM regime classification during known historical crisis periods."""
    print("\n" + "=" * 75)
    print("  AUDIT 1: HISTORICAL CRISIS EPOCH REGIME ALIGNMENT")
    print("=" * 75)

    dates = df_regimes.index

    for crisis in HISTORICAL_CRISES:
        mask = (dates >= pd.to_datetime(crisis["start"])) & (dates <= pd.to_datetime(crisis["end"]))
        df_c = df_regimes[mask]

        if len(df_c) == 0:
            print(f"  {crisis['name']}: No data rows in sample range.")
            continue

        regime_counts = df_c["regime_name"].value_counts()
        top_regime = regime_counts.index[0]
        top_pct = (regime_counts.iloc[0] / len(df_c)) * 100

        bear_count = sum(cnt for name, cnt in regime_counts.items() if "Bear" in name or "Stress" in name or "Late Cycle" in name)
        bear_pct = (bear_count / len(df_c)) * 100

        status = "PASSED" if bear_pct >= 50.0 else "WARNING (Low Crisis Detection)"
        print(f"\n  Epoch: {crisis['name']}")
        print(f"  Date Range : {crisis['start']} to {crisis['end']} ({len(df_c)} trading days)")
        print(f"  Dominant Regime  : '{top_regime}' ({top_pct:.1f}% of days)")
        print(f"  Total Bear/Stress/Late-Cycle Detection: {bear_pct:.1f}% [{status}]")

    print("=" * 75 + "\n")


def audit_baseline_threshold_comparison(df_regimes: pd.DataFrame):
    """Audit #2: Evaluates HMM vs simple heuristic rule baseline (VIX percentile > 0.7 OR Credit Spread > 3.5)."""
    print("\n" + "=" * 75)
    print("  AUDIT 2: HEURISTIC THRESHOLD RULE BASELINE VS HMM MODEL")
    print("=" * 75)

    vix_col = "vix_percentile" if "vix_percentile" in df_regimes.columns else None
    credit_col = "credit_spread" if "credit_spread" in df_regimes.columns else None

    if not vix_col:
        print("  Skipping: VIX percentile column not found.")
        return

    vix_vals = df_regimes[vix_col].values
    credit_vals = df_regimes[credit_col].values if credit_col else np.zeros(len(df_regimes))

    rule_bear_mask = (vix_vals > 0.65) | (credit_vals > 3.0)

    hmm_regime_names = df_regimes["regime_name"].values
    hmm_bear_mask = np.array(["Bear" in r or "Stress" in r for r in hmm_regime_names])

    agreement = np.mean(rule_bear_mask == hmm_bear_mask) * 100
    tp = np.sum(rule_bear_mask & hmm_bear_mask)
    fp = np.sum(~rule_bear_mask & hmm_bear_mask)
    fn = np.sum(rule_bear_mask & ~hmm_bear_mask)
    tn = np.sum(~rule_bear_mask & ~hmm_bear_mask)

    print(f"  Rule Baseline Definition: VIX Percentile > 65th percentile OR Credit Spread > 3.0")
    print(f"  HMM vs Baseline Overall Agreement : {agreement:.2f}%")
    print(f"  Confusion Matrix (Rule=Actual, HMM=Predicted):")
    print(f"    - True Positives  (Both Bear/Stress) : {tp:>5} days")
    print(f"    - False Positives (HMM Bear, Rule Normal): {fp:>5} days")
    print(f"    - False Negatives (HMM Normal, Rule Bear): {fn:>5} days")
    print(f"    - True Negatives  (Both Normal/Bull) : {tn:>5} days")
    print("=" * 75 + "\n")


def audit_feature_means_per_regime(predictor: HMMPredictor, X: np.ndarray, df_clean: pd.DataFrame):
    """Audit #3: Breakdown of empirical macro feature means per state."""
    print("\n" + "=" * 75)
    print("  AUDIT 3: EMPIRICAL FEATURE MEANS & STRESS SCORE BREAKDOWN")
    print("=" * 75)

    mapping = predictor.ensure_regime_mapping(X, df_clean)
    state_seq = predictor.model.predict(X)

    stats = []
    for s in range(predictor.n_states):
        mask = (state_seq == s)
        if not np.any(mask):
            continue
        n_obs = mask.sum()
        reg_name = get_regime_name(s, mapping)
        
        row = {"State": s, "Regime Name": reg_name[:32], "Obs Count": n_obs, "% Days": f"{(n_obs/len(X))*100:.1f}%"}
        for f in ["vix_percentile", "credit_spread", "yield_curve", "spy_returns", "inflation_z"]:
            if f in df_clean.columns:
                val = df_clean[f].values[mask].mean()
                if f == "spy_returns":
                    val *= 100
                row[f] = round(float(val), 3)
        stats.append(row)
        
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    print("=" * 75 + "\n")


def audit_seed_stability(X: np.ndarray, n_states: int = 7, n_seeds: int = 4):
    """Audit #4: Multi-seed stability test for HMM regime boundaries."""
    print("\n" + "=" * 75)
    print(f"  AUDIT 4: MULTI-SEED HMM STABILITY TEST ({n_seeds} Seeds for K={n_states})")
    print("=" * 75)

    base_model, _ = train_single_hmm(X, n_states=n_states, random_state=42, n_init=20, n_iter=50)
    base_seq = base_model.predict(X)

    agreements = []
    for seed in [10, 100, 2024, 777]:
        m_seed, _ = train_single_hmm(X, n_states=n_states, random_state=seed, n_init=20, n_iter=50)
        seq_seed = m_seed.predict(X)
        
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(base_seq, seq_seed)
        agreements.append(ari)
        print(f"  Seed {seed:>5} vs Seed 42 Baseline | Adjusted Rand Index (ARI): {ari:.4f} {'[HIGH STABILITY]' if ari > 0.6 else '[MODERATE]'}")

    mean_ari = np.mean(agreements)
    print(f"  {'-'*71}")
    print(f"  Mean Multi-Seed ARI Alignment: {mean_ari:.4f} {'[PRODUCTION STABLE]' if mean_ari > 0.55 else '[SENSITIVE TO SEED]'}")
    print("=" * 75 + "\n")


def main():
    X, feature_names, df_clean = retrieve_features(config.PARQUET_FILE, use_only_z_features=config.USE_ONLY_Z_FEATURES)
    predictor = HMMPredictor()
    df_regimes = predictor.get_regime_dataframe(X, df_clean)

    audit_crisis_alignment(df_regimes)
    audit_baseline_threshold_comparison(df_regimes)
    audit_feature_means_per_regime(predictor, X, df_clean)
    audit_seed_stability(X, n_states=predictor.n_states, n_seeds=4)


if __name__ == "__main__":
    main()
