"""
tests/audit_and_validate.py

Comprehensive Audit & Validation Suite for Gaussian HMM Market Regime Detection:
  1. Historical Crisis Date Alignment & Sanity Check (#5)
  2. Simple Threshold Rule Baseline Comparison & Confusion Matrix (#3)
  3. Feature Importance / Means Breakdown per Regime (#4)
  4. Multi-Seed Regime Label Stability Test (#2)
"""

from __future__ import annotations

import os
import sys
import io

# Ensure UTF-8 stdout encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from load import retrieve_features
from predict import HMMPredictor, get_regime_name, build_dynamic_regime_mapping
from train import train_single_hmm
import config

HISTORICAL_CRISES = [
    {"name": "2008 Global Financial Crisis (GFC)", "start": "2007-12-01", "end": "2009-06-30"},
    {"name": "2011 US Credit Downgrade / Euro Debt", "start": "2011-07-01", "end": "2011-10-31"},
    {"name": "2015-16 China Devaluation / Oil Collapse", "start": "2015-08-01", "end": "2016-02-28"},
    {"name": "2020 COVID-19 Liquidity Shock", "start": "2020-02-15", "end": "2020-04-30"},
    {"name": "2022 Fed Rate-Hike Inflation Bear Market", "start": "2022-01-01", "end": "2022-11-30"},
    {"name": "2023 SVB Regional Banking Shock", "start": "2023-03-01", "end": "2023-05-15"},
]


def audit_crisis_alignment(df_regimes: pd.DataFrame) -> Dict[str, Any]:
    """Audit #1: Sanity-check regime dates against known historical crises."""
    print("\n" + "=" * 75)
    print("  AUDIT 1: HISTORICAL CRISIS ALIGNMENT SANITY CHECK")
    print("=" * 75)

    results = {}
    for crisis in HISTORICAL_CRISES:
        c_start = pd.to_datetime(crisis["start"])
        c_end = pd.to_datetime(crisis["end"])
        mask = (df_regimes.index >= c_start) & (df_regimes.index <= c_end)
        sub_df = df_regimes[mask]
        
        n_days = len(sub_df)
        if n_days == 0:
            continue
            
        stress_bear_mask = sub_df["regime_name"].str.contains("Recessionary|Bear|Stress|Stagflation", case=False, regex=True)
        stress_days = stress_bear_mask.sum()
        pct_stress = (stress_days / n_days) * 100
        
        top_regime = sub_df["regime_name"].value_mode()[0] if hasattr(sub_df["regime_name"], "value_mode") else sub_df["regime_name"].value_counts().index[0]
        
        results[crisis["name"]] = {
            "total_days": n_days,
            "stress_days": stress_days,
            "pct_stress": pct_stress,
            "dominant_regime": top_regime
        }
        
        status = "✅ STRONG SIGNAL" if pct_stress >= 50 else ("⚠️ MODERATE SIGNAL" if pct_stress >= 25 else "❌ WEAK SIGNAL")
        print(f"  {crisis['name']:<42} | Stress Days: {stress_days:>3}/{n_days:<3} ({pct_stress:>5.1f}%) | {status}")
        print(f"    -> Dominant Regime: {top_regime}")

    print("=" * 75)
    return results


def audit_baseline_threshold_comparison(df_regimes: pd.DataFrame) -> Dict[str, Any]:
    """Audit #2: Compare HMM predictions against simple threshold rule baseline."""
    print("\n" + "=" * 75)
    print("  AUDIT 2: HMM VS SIMPLE THRESHOLD RULE BASELINE COMPARISON")
    print("=" * 75)

    # Define simple threshold baseline rule
    vix = df_regimes["vix_percentile"] if "vix_percentile" in df_regimes.columns else np.zeros(len(df_regimes))
    credit = df_regimes["credit_spread"] if "credit_spread" in df_regimes.columns else np.zeros(len(df_regimes))
    
    baseline_bear = (vix > 0.70) | (credit > 3.0)
    baseline_cat = np.where(baseline_bear, "BEAR", "NON-BEAR")

    hmm_cat = np.where(
        df_regimes["regime_name"].str.contains("Recessionary|Bear|Stress|Stagflation", case=False, regex=True),
        "BEAR", "NON-BEAR"
    )

    # Confusion matrix
    tp = sum((hmm_cat == "BEAR") & (baseline_cat == "BEAR"))
    fp = sum((hmm_cat == "BEAR") & (baseline_cat == "NON-BEAR"))
    fn = sum((hmm_cat == "NON-BEAR") & (baseline_cat == "BEAR"))
    tn = sum((hmm_cat == "NON-BEAR") & (baseline_cat == "NON-BEAR"))

    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = ((tp + tn) / len(df_regimes)) * 100

    print(f"  Threshold Rule Baseline Definition: BEAR if VIX Percentile > 70% OR Credit Spread > 3.0%")
    print(f"  {'-'*71}")
    print(f"  Total Market Days Evaluated : {len(df_regimes):,}")
    print(f"  HMM Detected BEAR Days      : {sum(hmm_cat == 'BEAR'):,} ({(sum(hmm_cat == 'BEAR')/len(df_regimes))*100:.1f}%)")
    print(f"  Baseline Rule BEAR Days     : {sum(baseline_cat == 'BEAR'):,} ({(sum(baseline_cat == 'BEAR')/len(df_regimes))*100:.1f}%)")
    print(f"  {'-'*71}")
    print(f"  Confusion Matrix (HMM vs Threshold Rule):")
    print(f"    True Positives  (Both BEAR)     : {tp:>5,}")
    print(f"    False Positives (HMM Bear only) : {fp:>5,}  (Multivariate sequential context)")
    print(f"    False Negatives (Rule Bear only): {fn:>5,}")
    print(f"    True Negatives  (Both Non-Bear) : {tn:>5,}")
    print(f"  {'-'*71}")
    print(f"  Agreement Accuracy : {accuracy:>6.1f}%")
    print(f"  Bear Precision     : {precision:>6.1f}%")
    print(f"  Bear Recall        : {recall:>6.1f}%")
    print(f"  Bear F1 Score      : {f1:>6.1f}%")
    print("=" * 75)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }


def audit_feature_means_per_regime(predictor: HMMPredictor, X: np.ndarray, df_clean: pd.DataFrame):
    """Audit #3: Surface feature importance & empirical means per regime."""
    print("\n" + "=" * 75)
    print("  AUDIT 3: EMPIRICAL FEATURE MEANS PER HMM REGIME STATE")
    print("=" * 75)

    mapping = predictor.ensure_regime_mapping(X, df_clean)
    state_seq = predictor.model.predict(X)
    
    feature_names = predictor.feature_names
    stats = []
    
    for s in range(predictor.n_states):
        mask = (state_seq == s)
        n_obs = mask.sum()
        reg_name = get_regime_name(s, mapping)
        
        row = {"State": s, "Regime Name": reg_name[:32], "Obs Count": n_obs, "% Days": f"{(n_obs/len(X))*100:.1f}%"}
        for f in ["vix_percentile", "credit_spread", "yield_curve", "spy_returns", "inflation_z"]:
            if f in df_clean.columns:
                val = df_clean[f].values[mask].mean()
                if f == "spy_returns":
                    val *= 100  # percentage
                row[f] = round(float(val), 3)
        stats.append(row)
        
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    print("=" * 75)


def audit_seed_stability(X: np.ndarray, n_states: int = 7, n_seeds: int = 5):
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
        
        # Calculate max overlap via optimal permutation matching
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
