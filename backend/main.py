import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd

# Add workspace root to sys.path to handle backend module imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend.app.quant.regime import load, predict
from backend.app.quant.analogue import dtw_engine
from backend.app.services import ai_analyst
from backend.app.core import config

from backend.app.quant.regime.load import retrieve_features
from backend.app.quant.regime.predict import HMMPredictor, get_regime_name
from backend.app.quant.analogue.dtw_engine import DTWEngine
from backend.app.services.ai_analyst import generate_ai_market_commentary

app = FastAPI(title="Market Regime Intelligence API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables initialized at startup
X: Optional[np.ndarray] = None
feature_names: List[str] = []
df_clean: Optional[pd.DataFrame] = None
df_regimes: Optional[pd.DataFrame] = None
predictor: Optional[HMMPredictor] = None

REGIME_COLORS = {
    "Goldilocks Expansion (Low Inflation)":         "#2ecc71",
    "Low Volatility Bull Market":                   "#27ae60",
    "Steady Growth / Mid-Cycle Expansion":          "#1abc9c",
    "Inflationary Bull / Growth Peak":              "#16a085",
    "Inflationary Expansion / Cyclical Peak":       "#f39c12",
    "Late Cycle / Inflationary Growth":             "#e67e22",
    "Market Consolidation / Neutral Growth":        "#3498db",
    "Sideways / Range-Bound Market":               "#2980b9",
    "Stagflationary Bear / Rate-Hike Stress":       "#e74c3c",
    "Recessionary Bear Market / Volatile Stress":   "#c0392b",
    "Recessionary Bear Market / High Stress":       "#922b21",
    "Transitional / Uncertain Regime":              "#95a5a6",
}


def load_pipeline():
    global X, feature_names, df_clean, df_regimes, predictor
    if X is None or predictor is None:
        X, feature_names, df_clean = retrieve_features(
            config.PARQUET_FILE, use_only_z_features=config.USE_ONLY_Z_FEATURES
        )
        predictor = HMMPredictor()
        df_regimes = predictor.get_regime_dataframe(X, df_clean)
        if "spy_returns" in df_regimes.columns:
            df_regimes["cumulative_market_index"] = (1 + df_regimes["spy_returns"]).cumprod() * 100.0
            df_regimes["rolling_30d_return"] = df_regimes["spy_returns"].rolling(30).sum() * 100.0
        else:
            df_regimes["cumulative_market_index"] = 100.0
            df_regimes["rolling_30d_return"] = 0.0


@app.on_event("startup")
def startup_event():
    load_pipeline()


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/dates")
def get_available_dates():
    load_pipeline()
    dates = df_clean.index.strftime("%Y-%m-%d").tolist()
    return {
        "dates": dates,
        "min_date": dates[0],
        "max_date": dates[-1],
        "total_days": len(dates)
    }


@app.get("/api/metadata")
def get_model_metadata():
    load_pipeline()
    mapping = predictor.ensure_regime_mapping(X, df_clean)
    meta = predictor.metadata.copy()
    meta["regime_mapping"] = mapping
    meta["regime_colors"] = REGIME_COLORS
    return meta


@app.get("/api/history")
def get_historical_data():
    load_pipeline()
    # Build list of records for chart consumption
    df_reset = df_regimes.reset_index()
    df_reset["date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")
    
    # Select columns to serialize
    cols_to_include = [
        "date", "cumulative_market_index", "rolling_30d_return", 
        "predicted_regime", "regime_name", "regime_confidence"
    ]
    for fn in feature_names:
        if fn in df_reset.columns and fn not in cols_to_include:
            cols_to_include.append(fn)

    # Replace inf/-inf with NaN, then NaN with None so JSON serialization doesn't fail
    df_output = df_reset[cols_to_include].replace([np.inf, -np.inf], np.nan)
    df_output = df_output.astype(object).where(pd.notnull(df_output), None)

    records = df_output.to_dict(orient="records")
    
    # Calculate regime frequency counts
    freq_counts = df_regimes["regime_name"].value_counts().to_dict()
    regime_frequencies = [
        {"name": k, "count": int(v), "color": REGIME_COLORS.get(k, "#95a5a6")}
        for k, v in freq_counts.items()
    ]

    return {
        "history": records,
        "feature_names": feature_names,
        "regime_frequencies": regime_frequencies,
        "regime_colors": REGIME_COLORS
    }


@app.get("/api/inference")
def get_date_inference(date: Optional[str] = Query(None)):
    load_pipeline()
    dates_list = df_clean.index.strftime("%Y-%m-%d").tolist()
    
    if not date or date not in dates_list:
        target_date_str = dates_list[-1]
    else:
        target_date_str = date

    date_info = predictor.predict_for_date(X, dates=df_clean.index, target_date_str=target_date_str, df_clean=df_clean)
    target_idx = date_info["target_idx"]
    macro_row = df_clean.iloc[target_idx].to_dict()

    # Radar values normalization
    radar_features = ["vix_percentile", "credit_spread", "yield_curve", "inflation_z", "fedfunds_z"]
    radar_labels = ["VIX Fear", "Credit Spread", "Yield Curve", "Inflation", "Fed Funds"]
    radar_data = []
    
    for f, label in zip(radar_features, radar_labels):
        v = float(macro_row.get(f, 0.0))
        col_vals = df_clean[f].values if f in df_clean.columns else np.array([0.0])
        v_min, v_max = float(col_vals.min()), float(col_vals.max())
        v_norm = float(np.clip((v - v_min) / (v_max - v_min + 1e-9), 0.0, 1.0))
        radar_data.append({"feature": label, "key": f, "value": round(v_norm, 4), "raw_value": round(v, 4)})

    # Convert state probability dicts to clean lists for React components
    state_probs_list = [
        {"name": k, "probability": round(v * 100, 2), "color": REGIME_COLORS.get(k, "#95a5a6")}
        for k, v in date_info["state_probabilities"].items()
    ]
    tomorrow_probs_list = [
        {"name": k, "probability": round(v * 100, 2), "color": REGIME_COLORS.get(k, "#95a5a6")}
        for k, v in date_info["tomorrow_probabilities"].items()
    ]

    return {
        "target_idx": target_idx,
        "date": date_info["date"],
        "current_regime": date_info["current_regime"],
        "current_regime_name": date_info["current_regime_name"],
        "current_regime_color": REGIME_COLORS.get(date_info["current_regime_name"], "#95a5a6"),
        "confidence_score": date_info["confidence_score"],
        "state_probabilities": state_probs_list,
        "tomorrow_regime": date_info["tomorrow_regime"],
        "tomorrow_regime_name": date_info["tomorrow_regime_name"],
        "tomorrow_regime_color": REGIME_COLORS.get(date_info["tomorrow_regime_name"], "#95a5a6"),
        "tomorrow_confidence": date_info["tomorrow_confidence"],
        "tomorrow_probabilities": tomorrow_probs_list,
        "persistence_prob": date_info["persistence_prob"],
        "transition_out_prob": date_info["transition_out_prob"],
        "top_transition_state": date_info["top_transition_state"],
        "top_transition_regime_name": date_info["top_transition_regime_name"],
        "top_transition_prob": date_info["top_transition_prob"],
        "has_next_day": date_info["has_next_day"],
        "actual_next_date": date_info["actual_next_date"],
        "actual_next_state": date_info["actual_next_state"],
        "actual_next_regime_name": date_info["actual_next_regime_name"],
        "actual_next_regime_color": REGIME_COLORS.get(date_info.get("actual_next_regime_name"), "#95a5a6") if date_info.get("actual_next_regime_name") else None,
        "actual_next_confidence": date_info["actual_next_confidence"],
        "actual_regime_changed": date_info["actual_regime_changed"],
        "macro_row": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in macro_row.items()},
        "radar_data": radar_data,
        "regime_mapping": date_info["regime_mapping"],
    }


@app.get("/api/dtw")
def get_dtw_matches(
    date: str = Query(...),
    window_size: int = Query(30, ge=15, le=90),
    top_k: int = Query(3, ge=1, le=5)
):
    load_pipeline()
    dates_list = df_clean.index.strftime("%Y-%m-%d").tolist()
    
    if date not in dates_list:
        target_date_str = dates_list[-1]
    else:
        target_date_str = date

    target_idx = dates_list.index(target_date_str)
    q_start_idx = max(0, target_idx - window_size)
    q_start_date_str = df_clean.index[q_start_idx].strftime("%Y-%m-%d")

    dtw_engine_obj = DTWEngine(window_size=window_size)
    spy_vals = df_clean["spy_returns"].values if "spy_returns" in df_clean.columns else None

    matches = dtw_engine_obj.find_similar_trajectories(
        X=X,
        dates=df_clean.index,
        regimes=df_regimes["predicted_regime"].values,
        top_n=top_k,
        query_start_date=q_start_date_str,
        query_end_date=target_date_str,
        spy_returns=spy_vals
    )

    mapping = predictor.ensure_regime_mapping(X, df_clean)
    query_vix = df_clean.iloc[q_start_idx: target_idx + 1]["vix_percentile"].values.tolist()

    formatted_matches = []
    for rank, m in enumerate(matches, 1):
        dominant_nm = mapping.get(m["dominant_regime"], get_regime_name(m["dominant_regime"])) if m["dominant_regime"] is not None else "N/A"
        hist_vix = df_clean.iloc[m["start_idx"]: m["end_idx"]]["vix_percentile"].values.tolist()
        
        formatted_matches.append({
            "rank": rank,
            "start_date": m["start_date"],
            "end_date": m["end_date"],
            "similarity_score": round(m["similarity_score"] * 100, 2),
            "normalized_distance": round(m["normalized_distance"], 4),
            "dominant_regime": m["dominant_regime"],
            "dominant_regime_name": dominant_nm,
            "dominant_regime_color": REGIME_COLORS.get(dominant_nm, "#95a5a6"),
            "forward_30d_return": round(m["forward_30d_return"], 2) if m.get("forward_30d_return") is not None else None,
            "vix_values": hist_vix
        })

    return {
        "query_start_date": q_start_date_str,
        "query_end_date": target_date_str,
        "window_size": window_size,
        "query_vix": query_vix,
        "matches": formatted_matches
    }


@app.get("/api/transition-matrix")
def get_transition_matrix():
    load_pipeline()
    trans_matrix = (predictor.model.transmat_ * 100.0).tolist()
    n_states = predictor.n_states
    mapping = predictor.ensure_regime_mapping(X, df_clean)

    states_labels = [mapping.get(i, get_regime_name(i)) for i in range(n_states)]
    short_labels = [s.split("/")[0].strip()[:22] for s in states_labels]

    tm = predictor.model.transmat_
    persistence_table = []
    for s in range(n_states):
        p_val = float(tm[s, s] * 100.0)
        avg_days = int(round(1.0 / (1.0 - tm[s, s]))) if tm[s, s] < 1.0 else 999
        reg_nm = mapping.get(s, get_regime_name(s))
        persistence_table.append({
            "state_id": s,
            "regime_name": reg_nm,
            "regime_color": REGIME_COLORS.get(reg_nm, "#95a5a6"),
            "daily_persistence": round(p_val, 1),
            "avg_days": avg_days
        })

    persistence_table.sort(key=lambda x: x["daily_persistence"], reverse=True)

    return {
        "transition_matrix": trans_matrix,
        "states_labels": states_labels,
        "short_labels": short_labels,
        "persistence_table": persistence_table
    }


@app.get("/api/ai-report")
def get_ai_report(date: str = Query(...)):
    load_pipeline()
    dates_list = df_clean.index.strftime("%Y-%m-%d").tolist()
    
    if date not in dates_list:
        target_date_str = dates_list[-1]
    else:
        target_date_str = date

    date_info = predictor.predict_for_date(X, dates=df_clean.index, target_date_str=target_date_str, df_clean=df_clean)
    target_idx = date_info["target_idx"]
    macro_row = df_clean.iloc[target_idx].to_dict()

    q_start_idx = max(0, target_idx - 30)
    q_start_date_str = df_clean.index[q_start_idx].strftime("%Y-%m-%d")

    dtw_engine_obj = DTWEngine(window_size=30)
    spy_vals = df_clean["spy_returns"].values if "spy_returns" in df_clean.columns else None

    matches = dtw_engine_obj.find_similar_trajectories(
        X=X,
        dates=df_clean.index,
        regimes=df_regimes["predicted_regime"].values,
        top_n=3,
        query_start_date=q_start_date_str,
        query_end_date=target_date_str,
        spy_returns=spy_vals
    )

    report = generate_ai_market_commentary(
        regime_info=date_info,
        macro_row=macro_row,
        dtw_matches=matches
    )

    return {
        "date": target_date_str,
        "report": report
    }