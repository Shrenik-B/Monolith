from __future__ import annotations

"""
dtw_engine.py

Purpose:
--------
Dynamic Time Warping (DTW) Engine for market trajectory similarity matching.
Compares a target market trajectory (e.g. specified by date range or latest 30-90 days) 
against rolling historical windows to identify top 3 similar historical market conditions.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from backend.app.quant.regime.load import retrieve_features
from backend.app.core import config

logger = logging.getLogger(__name__)


class DTWEngine:
    """
    Multivariate Dynamic Time Warping (DTW) Search Engine for Market Trajectories.
    """

    def __init__(self, window_size: int = config.DTW_WINDOW_SIZE, step_size: int = config.DTW_STEP_SIZE):
        self.window_size = window_size
        self.step_size = step_size

    def find_similar_trajectories(
        self,
        X: np.ndarray,
        dates: pd.DatetimeIndex,
        regimes: Optional[np.ndarray] = None,
        top_n: int = config.DTW_TOP_N,
        query_start_date: Optional[str] = None,
        query_end_date: Optional[str] = None,
        spy_returns: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Finds the top N most similar historical trajectory windows matching the query market window.
        
        Parameters:
            query_start_date (str, optional): 'YYYY-MM-DD' start date for custom query window.
            query_end_date (str, optional): 'YYYY-MM-DD' end date for custom query window.
        """
        n_samples, n_features = X.shape

        if query_start_date and query_end_date:
            dates_str = dates.strftime("%Y-%m-%d").tolist()
            try:
                q_start_idx = dates_str.index(query_start_date)
                q_end_idx = dates_str.index(query_end_date) + 1
            except ValueError:
                mask = (dates >= pd.to_datetime(query_start_date)) & (dates <= pd.to_datetime(query_end_date))
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    q_start_idx = n_samples - self.window_size
                    q_end_idx = n_samples
                else:
                    q_start_idx = indices[0]
                    q_end_idx = indices[-1] + 1
            
            if q_end_idx - q_start_idx < 5:
                q_start_idx = max(0, q_end_idx - self.window_size)
                
            query_window = X[q_start_idx:q_end_idx]
            query_len = q_end_idx - q_start_idx
        else:
            q_start_idx = n_samples - self.window_size
            q_end_idx = n_samples
            query_window = X[q_start_idx:q_end_idx]
            query_len = self.window_size

        matches = []
        hist_window_len = query_len

        for start_idx in range(0, n_samples - hist_window_len + 1, self.step_size):
            end_idx = start_idx + hist_window_len

            if end_idx > q_start_idx:
                continue

            hist_window = X[start_idx:end_idx]

            distance, _ = fastdtw(query_window, hist_window, dist=euclidean)
            norm_distance = distance / hist_window_len
            similarity_score = 1.0 / (1.0 + norm_distance)

            match_start_date = dates[start_idx].strftime("%Y-%m-%d")
            match_end_date = dates[end_idx - 1].strftime("%Y-%m-%d")

            dominant_regime = None
            if regimes is not None:
                hist_regimes = regimes[start_idx:end_idx]
                counts = np.bincount(hist_regimes)
                dominant_regime = int(np.argmax(counts))

            forward_return = None
            if spy_returns is not None and end_idx + 30 <= n_samples:
                forward_return = float(np.sum(spy_returns[end_idx : end_idx + 30]) * 100)

            matches.append({
                "start_date": match_start_date,
                "end_date": match_end_date,
                "dtw_distance": float(distance),
                "normalized_distance": float(norm_distance),
                "similarity_score": float(similarity_score),
                "dominant_regime": dominant_regime,
                "forward_30d_return": forward_return,
                "start_idx": start_idx,
                "end_idx": end_idx,
            })

        matches_sorted = sorted(matches, key=lambda x: x["normalized_distance"])

        distinct_matches = []
        selected_indices = []

        for m in matches_sorted:
            overlap = False
            for s_start, s_end in selected_indices:
                if not (m["end_idx"] <= s_start or m["start_idx"] >= s_end):
                    overlap = True
                    break
            if not overlap:
                distinct_matches.append(m)
                selected_indices.append((m["start_idx"], m["end_idx"]))
                if len(distinct_matches) == top_n:
                    break

        return distinct_matches


if __name__ == "__main__":
    X, feature_names, df_clean = retrieve_features(config.PARQUET_FILE, use_only_z_features=config.USE_ONLY_Z_FEATURES)
    dtw = DTWEngine(window_size=config.DTW_WINDOW_SIZE)
    matches = dtw.find_similar_trajectories(X, dates=df_clean.index, top_n=config.DTW_TOP_N)

    print("\n=======================================================")
    print("      DTW HISTORICAL MARKET TRAJECTORY MATCHING       ")
    print("=======================================================")
    print(f" Target Window Size : {config.DTW_WINDOW_SIZE} trading days")
    print(f" Latest Window      : {df_clean.index[-config.DTW_WINDOW_SIZE].strftime('%Y-%m-%d')} to {df_clean.index[-1].strftime('%Y-%m-%d')}")
    print("-------------------------------------------------------")
    print(" Top Matching Historical Periods:")
    for rank, m in enumerate(matches, 1):
        print(f" #{rank}: Period {m['start_date']} -> {m['end_date']} | "
              f"Similarity: {m['similarity_score']*100:.2f}% (Dist: {m['normalized_distance']:.4f})")
    print("=======================================================\n")
