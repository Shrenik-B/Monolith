"""
pipeline.py

Purpose:
--------
End-to-End Market Regime Detection MVP Pipeline.
Executes complete sequence:
  1. Load Data (Parquet dataset extraction)
  2. K-Means++ Initialization & Gaussian HMM Baum-Welch Training
  3. Model Selection (AIC / BIC automated selection)
  4. Model Persistence (save model.pkl & model_metadata.json)
  5. Inference Engine (predict latest regime, state confidence, transitions)
  6. DTW Similarity Engine (find top historical trajectory matches for explainability)
"""

from __future__ import annotations

import logging
import time

from load import retrieve_features
from train import select_best_hmm
from model_persistence import save_model_and_metadata, load_model_and_metadata
from predict import HMMPredictor
from dtw_engine import DTWEngine
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    start_time = time.time()
    print("==========================================================================")
    print("      STARTING END-TO-END MARKET REGIME MVP PIPELINE                      ")
    print("==========================================================================")

    # STEP 1: Load Data
    print("\n[STEP 1/6] Loading & Inspecting Feature Dataset...")
    X, feature_names, df_clean = retrieve_features(
        file_name=config.PARQUET_FILE,
        use_only_z_features=config.USE_ONLY_Z_FEATURES
    )

    # STEP 2 & 3: K-Means++ Initialization, Gaussian HMM Training & Model Selection
    print("\n[STEP 2 & 3/6] Running K-Means++ Init, Gaussian HMM Training & Model Selection...")
    best_model, best_metrics, summary_df = select_best_hmm(
        X=X,
        candidate_states=config.CANDIDATE_STATES,
        covariance_type=config.COVARIANCE_TYPE,
        random_state=config.RANDOM_STATE
    )

    # STEP 4: Model Persistence
    print("\n[STEP 4/6] Persisting Best HMM Model & Metadata...")
    save_model_and_metadata(
        model=best_model,
        metrics=best_metrics,
        feature_names=feature_names,
        model_path=config.MODEL_FILE,
        metadata_path=config.METADATA_FILE,
        summary_df=summary_df
    )

    # STEP 5: Inference Engine
    print("\n[STEP 5/6] Executing Inference Engine on Latest Market Observations...")
    predictor = HMMPredictor(
        model_path=config.MODEL_FILE,
        metadata_path=config.METADATA_FILE
    )
    inference_res = predictor.predict_for_index(X, dates=df_clean.index, target_idx=-1, df_clean=df_clean)

    print("\n--------------------------------------------------------------------------")
    print("                       CURRENT MARKET INFERENCE RESULTS                   ")
    print("--------------------------------------------------------------------------")
    print(f" Date                     : {inference_res['date']}")
    print(f" Current Market Regime    : State {inference_res['current_regime']}")
    print(f" Regime Confidence Score  : {inference_res['confidence_score'] * 100:.2f}%")
    print("\n State Posterior Probabilities:")
    for state, prob in inference_res['state_probabilities'].items():
        print(f"   - {state}: {prob * 100:.2f}%")
    print("\n Next-Day State Transition Probabilities:")
    for to_state, prob in inference_res['transition_probabilities'].items():
        print(f"   - {to_state}: {prob * 100:.2f}%")

    # STEP 6: DTW Historical Trajectory Similarity Engine
    print("\n[STEP 6/6] Running DTW Historical Trajectory Matching Engine...")
    dtw_engine = DTWEngine(
        window_size=config.DTW_WINDOW_SIZE,
        step_size=config.DTW_STEP_SIZE
    )
    spy_vals = df_clean["spy_returns"].values if "spy_returns" in df_clean.columns else None
    matches = dtw_engine.find_similar_trajectories(
        X=X,
        dates=df_clean.index,
        regimes=inference_res['full_state_sequence'],
        top_n=config.DTW_TOP_N,
        spy_returns=spy_vals
    )

    print("\n--------------------------------------------------------------------------")
    print("             TOP HISTORICAL MARKET TRAJECTORY MATCHES (EXPLAINABILITY)    ")
    print("--------------------------------------------------------------------------")
    print(f" Latest Market Trajectory Window: {df_clean.index[-config.DTW_WINDOW_SIZE].strftime('%Y-%m-%d')} to {df_clean.index[-1].strftime('%Y-%m-%d')}")
    for rank, m in enumerate(matches, 1):
        regime_str = f"State {m['dominant_regime']}" if m['dominant_regime'] is not None else "N/A"
        print(f" Match #{rank}: Period [{m['start_date']} -> {m['end_date']}] | "
              f"Similarity: {m['similarity_score']*100:.2f}% | "
              f"Dominant Historical Regime: {regime_str}")
    print("--------------------------------------------------------------------------")

    elapsed = time.time() - start_time
    print(f"\nPipeline completed successfully in {elapsed:.2f} seconds.")
    print("==========================================================================\n")


if __name__ == "__main__":
    run_pipeline()
