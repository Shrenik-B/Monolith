"""
predict.py

Purpose:
--------
Inference engine for loading trained Gaussian HMM models and predicting market regimes,
human-readable regime names, posterior state probabilities, tomorrow's forecasted regime, 
and transition matrices for any target market date.

Empirical Auto-Mapping:
Dynamically computes human-readable regime names based on empirical feature means 
(VIX percentile, credit spread, SPY returns, inflation) to ensure high-stress crash periods 
are accurately labeled as Recessionary Bear Markets regardless of HMM state permutation.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from model_persistence import load_model_and_metadata
from load import retrieve_features
import config

logger = logging.getLogger(__name__)

# Fallback Regime Names
REGIME_MAPPINGS = {
    0: "Late Cycle / Inflationary Growth",
    1: "Goldilocks Expansion (Low Inflation)",
    2: "Low Volatility Bull Market",
    3: "Inflationary Expansion / Cyclical Peak",
    4: "Recessionary Bear Market / High Stress",
    5: "Market Consolidation / Neutral Growth",
}


def build_dynamic_regime_mapping(
    model: Any, X: np.ndarray, feature_names: list[str], df_clean: pd.DataFrame
) -> Dict[int, str]:
    """
    Dynamically assigns human-readable economic regime names to HMM hidden state IDs
    based on empirical cluster statistics.

    Uses 6 macro signals: VIX percentile, credit spread, SPY returns, inflation z-score,
    yield curve (10Y-2Y), and fed funds rate z-score.

    Mapping tiers (descending stress):
      Tier 1 (max stress)   -> Recessionary Bear Market / High Stress
      Tier 2 (2nd stress)   -> Recessionary Bear / Volatile Stress  OR
                               Stagflationary Bear / Rate-Hike Stress  OR
                               Late Cycle / Inflationary Growth
      Tier 3+ (lower stress) -> Bull / Goldilocks / Inflationary Expansion / Neutral
    """
    state_sequence = model.predict(X)
    n_states = len(np.unique(state_sequence))

    def get_feature(col_name: str) -> np.ndarray:
        if df_clean is not None and col_name in df_clean.columns:
            return df_clean[col_name].values.astype(float)
        elif col_name in feature_names:
            idx = feature_names.index(col_name)
            if idx < X.shape[1]:
                return X[:, idx].astype(float)
        return np.zeros(len(X))

    vix_vals       = get_feature("vix_percentile")
    credit_vals    = get_feature("credit_spread")
    spy_vals       = get_feature("spy_returns")
    inflation_vals = get_feature("inflation_z")
    yc_vals        = get_feature("yield_curve")       # Negative = inverted = recession signal
    ff_vals        = get_feature("fedfunds_z")        # High = tight monetary policy

    stats = {}
    for s in range(n_states):
        mask = (state_sequence == s)
        if not np.any(mask):
            stats[s] = {"vix": 0.5, "credit": 2.0, "spy": 0.0, "inf": 0.0,
                        "yc": 0.5, "ff": 0.0, "stress": 0.0, "count": 0}
            continue
        v_m  = float(np.mean(vix_vals[mask]))
        c_m  = float(np.mean(credit_vals[mask]))
        s_m  = float(np.mean(spy_vals[mask]))
        i_m  = float(np.mean(inflation_vals[mask]))
        yc_m = float(np.mean(yc_vals[mask]))
        ff_m = float(np.mean(ff_vals[mask]))
        # Composite stress score (higher = more stressed):
        #   VIX percentile: x3.5 (primary panic indicator)
        #   Credit spread:  x1.5 (credit market health)
        #   SPY returns:    x150 negative (bear markets have negative daily returns)
        #   Yield curve:    inverted YC (negative value) adds stress
        #   Fed funds:      high rates add mild stress
        stress = (v_m * 3.5) + (c_m * 1.5) - (s_m * 150.0) + (max(-yc_m, 0) * 1.0) + (max(ff_m, 0) * 0.4)
        stats[s] = {"vix": v_m, "credit": c_m, "spy": s_m, "inf": i_m,
                    "yc": yc_m, "ff": ff_m, "stress": stress, "count": int(np.sum(mask))}

    sorted_by_stress = sorted(stats.keys(), key=lambda s: stats[s]["stress"], reverse=True)
    mapping = {}

    # ── Tier 1: Most stressed state ──────────────────────────────────────────
    if len(sorted_by_stress) > 0:
        mapping[sorted_by_stress[0]] = "Recessionary Bear Market / High Stress"

    # ── Tier 2: Second most stressed state ───────────────────────────────────
    if len(sorted_by_stress) > 1:
        s1 = sorted_by_stress[1]
        st = stats[s1]
        if st["vix"] > 0.55 or st["credit"] > 2.8:
            mapping[s1] = "Recessionary Bear Market / Volatile Stress"
        elif st["spy"] < -0.0005 and (st["yc"] < 0.2 or st["ff"] > 0.5):
            mapping[s1] = "Stagflationary Bear / Rate-Hike Stress"
        else:
            mapping[s1] = "Late Cycle / Inflationary Growth"

    # ── Tier 3: Third most stressed (only for K>=5) ───────────────────────────
    if len(sorted_by_stress) > 2:
        s2 = sorted_by_stress[2]
        st = stats[s2]
        # Only assign a bear/stress name if it genuinely qualifies
        if s2 not in mapping:
            if (st["vix"] > 0.45 or st["credit"] > 2.5) and st["spy"] < 0.0:
                mapping[s2] = "Late Cycle / Inflationary Growth"

    # ── Tier 4+: Expansion / Bull / Neutral states ───────────────────────────
    remaining = [s for s in sorted_by_stress if s not in mapping]
    remaining_by_vix = sorted(remaining, key=lambda s: (stats[s]["vix"], -stats[s]["spy"]))

    # Lowest VIX + best returns = Low Volatility Bull Market
    if len(remaining_by_vix) > 0:
        s_bull = remaining_by_vix[0]
        mapping[s_bull] = "Low Volatility Bull Market"

    remaining2 = [s for s in remaining_by_vix if s not in mapping]
    remaining_by_inf = sorted(remaining2, key=lambda s: stats[s]["inf"], reverse=True)

    if len(remaining_by_inf) > 0:
        mapping[remaining_by_inf[0]] = "Inflationary Expansion / Cyclical Peak"
    if len(remaining_by_inf) > 1:
        mapping[remaining_by_inf[-1]] = "Goldilocks Expansion (Low Inflation)"
    if len(remaining_by_inf) > 2:
        # Middle inflation state = steady growth
        mapping[remaining_by_inf[1]] = "Steady Growth / Mid-Cycle Expansion"

    # Any remaining states = unique fallback names (no duplicates)
    fallback_names = [
        "Market Consolidation / Neutral Growth",
        "Sideways / Range-Bound Market",
        "Transitional / Uncertain Regime",
    ]
    fb_idx = 0
    for s in range(n_states):
        if s not in mapping:
            mapping[s] = fallback_names[min(fb_idx, len(fallback_names) - 1)]
            fb_idx += 1

    return mapping


def get_regime_name(state_id: int, custom_mapping: Optional[Dict[int, str]] = None) -> str:
    """Returns human-readable regime name for a given state integer ID."""
    if custom_mapping and state_id in custom_mapping:
        return custom_mapping[state_id]
    return REGIME_MAPPINGS.get(state_id, f"Regime State {state_id}")


class HMMPredictor:
    """
    Inference Engine for Gaussian HMM Market Regime Prediction & Forecasts.
    """

    def __init__(self, model_path: str = config.MODEL_FILE, metadata_path: str = config.METADATA_FILE):
        self.model, self.metadata = load_model_and_metadata(model_path, metadata_path)
        self.n_states = self.metadata["n_states"]
        self.feature_names = self.metadata["feature_names"]
        self.regime_mapping: Optional[Dict[int, str]] = None
        self._cached_X_id: Optional[int] = None
        self._cached_state_seq: Optional[np.ndarray] = None
        self._cached_posterior: Optional[np.ndarray] = None

    def _get_predictions(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Caches state sequence and posterior probabilities to avoid recomputation on every call."""
        x_id = id(X)
        if (
            self._cached_X_id != x_id
            or self._cached_state_seq is None
            or self._cached_posterior is None
            or len(self._cached_state_seq) != len(X)
        ):
            self._cached_state_seq = self.model.predict(X)
            self._cached_posterior = self.model.predict_proba(X)
            self._cached_X_id = x_id
        return self._cached_state_seq, self._cached_posterior

    def ensure_regime_mapping(self, X: np.ndarray, df_clean: Optional[pd.DataFrame] = None) -> Dict[int, str]:
        """Ensures dynamic empirical regime mapping is computed."""
        if self.regime_mapping is None:
            if df_clean is None:
                # Mock minimal dataframe from X if not provided
                df_clean = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
            self.regime_mapping = build_dynamic_regime_mapping(self.model, X, self.feature_names, df_clean)
        return self.regime_mapping

    def predict_for_index(
        self, X: np.ndarray, dates: pd.DatetimeIndex, target_idx: int = -1, df_clean: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Predicts market regime, confidence score, tomorrow's forecast, and transition probabilities for a specific index.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        mapping = self.ensure_regime_mapping(X, df_clean)

        n_samples = len(X)
        if target_idx < 0:
            target_idx = n_samples + target_idx

        target_idx = max(0, min(n_samples - 1, target_idx))

        # 1. Retrieve Viterbi hidden state sequence & posterior probabilities (cached)
        state_sequence, posterior_probs = self._get_predictions(X)

        # 2. Extract results for target_idx
        current_state = int(state_sequence[target_idx])
        current_probs = posterior_probs[target_idx]
        confidence_score = float(np.max(current_probs))
        current_regime_name = get_regime_name(current_state, mapping)
        current_date = dates[target_idx].strftime("%Y-%m-%d")

        # 3. Transition probabilities from current state
        transition_vector = self.model.transmat_[current_state]

        # 4. Forecast Tomorrow's Market Regime (1-day Markov forward projection)
        tomorrow_prob_vector = np.dot(current_probs, self.model.transmat_)
        tomorrow_state = int(np.argmax(tomorrow_prob_vector))
        tomorrow_regime_name = get_regime_name(tomorrow_state, mapping)
        tomorrow_confidence = float(np.max(tomorrow_prob_vector))

        # 5. Off-diagonal Transition Target Candidate (if a regime shift occurs)
        off_diag_probs = transition_vector.copy()
        off_diag_probs[current_state] = -1.0
        top_transition_state = int(np.argmax(off_diag_probs))
        top_transition_regime_name = get_regime_name(top_transition_state, mapping)
        top_transition_prob = float(transition_vector[top_transition_state])
        persistence_prob = float(transition_vector[current_state])
        transition_out_prob = float(1.0 - persistence_prob)

        # 6. Ground Truth Realized Next-Day State (available for historical dates)
        has_next_day = (target_idx < n_samples - 1)
        if has_next_day:
            actual_next_idx = target_idx + 1
            actual_next_date = dates[actual_next_idx].strftime("%Y-%m-%d")
            actual_next_state = int(state_sequence[actual_next_idx])
            actual_next_regime_name = get_regime_name(actual_next_state, mapping)
            actual_next_confidence = float(np.max(posterior_probs[actual_next_idx]))
            actual_regime_changed = (actual_next_state != current_state)
        else:
            actual_next_date = None
            actual_next_state = None
            actual_next_regime_name = None
            actual_next_confidence = None
            actual_regime_changed = False

        result = {
            "target_idx": target_idx,
            "date": current_date,
            "current_regime": current_state,
            "current_regime_name": current_regime_name,
            "confidence_score": confidence_score,
            "state_probabilities": {
                get_regime_name(i, mapping): float(current_probs[i]) for i in range(self.n_states)
            },
            "raw_state_probs": current_probs,
            "transition_probabilities": {
                get_regime_name(j, mapping): float(transition_vector[j]) for j in range(self.n_states)
            },
            "tomorrow_regime": tomorrow_state,
            "tomorrow_regime_name": tomorrow_regime_name,
            "tomorrow_confidence": tomorrow_confidence,
            "tomorrow_probabilities": {
                get_regime_name(j, mapping): float(tomorrow_prob_vector[j]) for j in range(self.n_states)
            },
            # Realized Next-Day Ground Truth (for historical dates)
            "has_next_day": has_next_day,
            "actual_next_date": actual_next_date,
            "actual_next_state": actual_next_state,
            "actual_next_regime_name": actual_next_regime_name,
            "actual_next_confidence": actual_next_confidence,
            "actual_regime_changed": actual_regime_changed,
            # Enhanced Transition Candidate Metrics
            "persistence_prob": persistence_prob,
            "transition_out_prob": transition_out_prob,
            "top_transition_state": top_transition_state,
            "top_transition_regime_name": top_transition_regime_name,
            "top_transition_prob": top_transition_prob,
            # Full sequence references
            "full_state_sequence": state_sequence,
            "full_posterior_probs": posterior_probs,
            "regime_mapping": mapping,
        }

        return result

    def predict_for_date(
        self, X: np.ndarray, dates: pd.DatetimeIndex, target_date_str: str, df_clean: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Predicts market regime for a specific date string ('YYYY-MM-DD').
        """
        dates_str_list = dates.strftime("%Y-%m-%d").tolist()
        if target_date_str in dates_str_list:
            idx = dates_str_list.index(target_date_str)
        else:
            target_dt = pd.to_datetime(target_date_str)
            idx = int(np.argmin(np.abs(dates - target_dt)))

        return self.predict_for_index(X, dates, target_idx=idx, df_clean=df_clean)

    def get_regime_dataframe(self, X: np.ndarray, df_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Appends predicted state regime IDs, regime names, and confidence scores to DataFrame.
        """
        mapping = self.ensure_regime_mapping(X, df_clean)
        state_sequence, posterior_probs = self._get_predictions(X)
        confidence_scores = np.max(posterior_probs, axis=1)

        result_df = df_clean.copy()
        result_df["predicted_regime"] = state_sequence
        result_df["regime_name"] = [get_regime_name(s, mapping) for s in state_sequence]
        result_df["regime_confidence"] = confidence_scores

        for state in range(self.n_states):
            reg_name = get_regime_name(state, mapping)
            result_df[f"prob_{reg_name}"] = posterior_probs[:, state]

        return result_df


if __name__ == "__main__":
    X, feature_names, df_clean = retrieve_features(config.PARQUET_FILE, use_only_z_features=config.USE_ONLY_Z_FEATURES)
    predictor = HMMPredictor()
    latest_info = predictor.predict_for_index(X, dates=df_clean.index, target_idx=-1, df_clean=df_clean)

    print("\n=======================================================")
    print("           HMM MARKET REGIME INFERENCE                ")
    print("=======================================================")
    print(f" Observation Date          : {latest_info['date']}")
    print(f" Current Market Regime     : {latest_info['current_regime_name']} (State {latest_info['current_regime']})")
    print(f" Confidence Score          : {latest_info['confidence_score'] * 100:.2f}%")
    print(f" Tomorrow's Forecast Regime: {latest_info['tomorrow_regime_name']} ({latest_info['tomorrow_confidence']*100:.2f}%)")
    print("-------------------------------------------------------")
    print(" Dynamic Empirical Regime Mapping:")
    for k, v in latest_info['regime_mapping'].items():
        print(f"   - State {k} => {v}")
    print("=======================================================\n")
