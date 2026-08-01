"""
tests/evaluate.py

Purpose:
--------
Formal Train / Validate / Test evaluation of the Gaussian HMM Market Regime model.
"""

from __future__ import annotations

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import logging

from load import retrieve_features
from train import select_best_hmm, train_single_hmm
from predict import HMMPredictor, build_dynamic_regime_mapping, get_regime_name
import config

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

def stress_category(name: str) -> str:
    n = name.lower()
    if "recessionary" in n or "bear" in n or "stress" in n or "stagflation" in n:
        return "BEAR"
    elif "goldilocks" in n or ("low volatility" in n and "bull" in n) or "bull" in n:
        return "BULL"
    elif "inflation" in n or "cyclical" in n or "late cycle" in n:
        return "INFLATIONARY"
    return "NEUTRAL"

def compute_regime_stability(regime_seq: list) -> float:
    if len(regime_seq) < 2:
        return 100.0
    same = sum(1 for a, b in zip(regime_seq[:-1], regime_seq[1:]) if a == b)
    return same / (len(regime_seq) - 1) * 100.0

def evaluate_on_split(
    model, X_split: np.ndarray, df_split: pd.DataFrame, feature_names: list,
    mapping: dict, ref_predictor: HMMPredictor, X_full: np.ndarray,
    df_full: pd.DataFrame, split_name: str,
):
    n = len(X_split)
    if n == 0:
        return {}

    log_likelihood = model.score(X_split)
    ll_per_obs = log_likelihood / n

    wf_states = model.predict(X_split)
    wf_regime_names = [get_regime_name(s, mapping) for s in wf_states]
    wf_categories = [stress_category(r) for r in wf_regime_names]

    ref_dates = df_split.index
    ref_state_seq = ref_predictor.model.predict(X_full)
    ref_mapping = ref_predictor.ensure_regime_mapping(X_full, df_full)

    full_dates = df_full.index
    split_indices = [np.where(full_dates == d)[0][0] for d in ref_dates if d in full_dates]

    ref_regime_names = [get_regime_name(ref_state_seq[i], ref_mapping) for i in split_indices]
    ref_categories = [stress_category(r) for r in ref_regime_names]

    n_aligned = min(len(wf_categories), len(ref_categories))
    exact_matches = sum(1 for a, b in zip(wf_regime_names[:n_aligned], ref_regime_names[:n_aligned]) if a == b)
    cat_matches = sum(1 for a, b in zip(wf_categories[:n_aligned], ref_categories[:n_aligned]) if a == b)

    exact_acc = exact_matches / n_aligned * 100 if n_aligned > 0 else 0.0
    cat_acc = cat_matches / n_aligned * 100 if n_aligned > 0 else 0.0
    stability = compute_regime_stability(wf_regime_names)

    cats = ["BEAR", "BULL", "INFLATIONARY", "NEUTRAL"]
    cat_precision = {}
    for cat in cats:
        predicted_as_cat = [i for i, c in enumerate(wf_categories[:n_aligned]) if c == cat]
        if len(predicted_as_cat) == 0:
            cat_precision[cat] = {"precision": 0.0, "n": 0}
        else:
            correct = sum(1 for i in predicted_as_cat if ref_categories[i] == cat)
            cat_precision[cat] = {
                "precision": correct / len(predicted_as_cat) * 100,
                "n": len(predicted_as_cat),
            }

    from collections import Counter
    return {
        "split": split_name, "n_obs": n,
        "date_start": df_split.index[0].strftime("%Y-%m-%d"),
        "date_end": df_split.index[-1].strftime("%Y-%m-%d"),
        "log_likelihood": log_likelihood, "ll_per_obs": ll_per_obs,
        "exact_accuracy": exact_acc, "category_accuracy": cat_acc,
        "regime_stability": stability, "n_aligned": n_aligned,
        "wf_dist": dict(Counter(wf_regime_names)),
        "ref_dist": dict(Counter(ref_regime_names[:n_aligned])),
        "cat_precision": cat_precision,
    }

def print_metrics(m: dict):
    print(f"\n  {'='*72}")
    print(f"  {m['split']} SET  |  {m['date_start']} -> {m['date_end']}  |  {m['n_obs']:,} observations")
    print(f"  {'='*72}")
    print(f"  {'METRIC':<45} {'VALUE':>20}")
    print(f"  {'-'*67}")
    print(f"  {'Log-Likelihood (total)':<45} {m['log_likelihood']:>20.2f}")
    print(f"  {'Log-Likelihood per observation':<45} {m['ll_per_obs']:>20.4f}")
    print(f"  {'Exact Regime Accuracy vs Ref Model':<45} {m['exact_accuracy']:>19.1f}%")
    print(f"  {'Category Accuracy vs Ref Model':<45} {m['category_accuracy']:>19.1f}%")
    print(f"  {'Regime Stability (no-switch days)':<45} {m['regime_stability']:>19.1f}%")
    print(f"  {'-'*67}")
    print(f"  Per-Category Precision (Walk-Forward vs Reference):")
    for cat, info in m["cat_precision"].items():
        bar = "#" * int(info["precision"] / 5) if info["n"] > 0 else ""
        print(f"    {cat:<15} {info['precision']:>6.1f}%  [{bar:<20}]  (n={info['n']})")

def main():
    print("\n" + "=" * 72)
    print("  TRAIN / VALIDATE / TEST TEMPORAL EVALUATION")
    print("  Gaussian HMM Market Regime Model -- Version 1")
    print("=" * 72)

    X_full, feature_names, df_full = retrieve_features(
        config.PARQUET_FILE, use_only_z_features=config.USE_ONLY_Z_FEATURES
    )
    n_total = len(X_full)

    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)
    n_test  = n_total - n_train - n_val

    X_train, df_train = X_full[:n_train], df_full.iloc[:n_train]
    X_val, df_val     = X_full[n_train: n_train + n_val], df_full.iloc[n_train: n_train + n_val]
    X_test, df_test   = X_full[n_train + n_val:], df_full.iloc[n_train + n_val:]

    print(f"\n  [STEP 2] Temporal Split:")
    print(f"  TRAIN: {n_train:,} | VAL: {n_val:,} | TEST: {n_test:,}")

    model, metrics, summary_df = select_best_hmm(
        X=X_train, candidate_states=config.CANDIDATE_STATES,
        covariance_type=config.COVARIANCE_TYPE, random_state=config.RANDOM_STATE,
    )
    best_k = int(metrics["n_states"])
    mapping = build_dynamic_regime_mapping(model, X_train, feature_names, df_train)

    try:
        ref_predictor = HMMPredictor(config.MODEL_FILE, config.METADATA_FILE)
        ref_predictor.ensure_regime_mapping(X_full, df_full)
    except Exception as e:
        print(f"  [WARNING] Reference model unavailable: {e}")
        return

    m_tr = evaluate_on_split(model, X_train, df_train, feature_names, mapping, ref_predictor, X_full, df_full, "TRAIN")
    m_va = evaluate_on_split(model, X_val, df_val, feature_names, mapping, ref_predictor, X_full, df_full, "VALIDATE")
    m_te = evaluate_on_split(model, X_test, df_test, feature_names, mapping, ref_predictor, X_full, df_full, "TEST")

    for m in [m_tr, m_va, m_te]:
        if m: print_metrics(m)

if __name__ == "__main__":
    main()
