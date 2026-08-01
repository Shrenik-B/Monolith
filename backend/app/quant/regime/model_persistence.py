from __future__ import annotations

"""
model_persistence.py

Purpose:
--------
Provides model persistence functions to save and load trained Gaussian HMM models
along with full metadata (AIC, BIC, feature list, states, training date).
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from hmmlearn.hmm import GaussianHMM
import pandas as pd

from backend.app.core import config

logger = logging.getLogger(__name__)


def save_model_and_metadata(
    model: GaussianHMM,
    metrics: Dict[str, Any],
    feature_names: list[str],
    model_path: str = config.MODEL_FILE,
    metadata_path: str = config.METADATA_FILE,
    summary_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Saves trained HMM model binary (.pkl), timestamped backup, metadata (.json), and model selection history (.csv).
    """
    # Ensure target directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # 1. Save primary model binary and timestamped backup
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved trained HMM model to '{model_path}'.")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(os.path.dirname(model_path), f"model_{timestamp_str}.pkl")
    try:
        with open(versioned_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved timestamped model backup to '{versioned_path}'.")
    except Exception as e:
        logger.warning(f"Could not save versioned model backup: {e}")

    # 2. Build metadata dictionary
    metadata = {
        "training_timestamp": datetime.now().isoformat(),
        "n_states": int(model.n_components),
        "covariance_type": model.covariance_type,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "metrics": {
            "log_likelihood": float(metrics.get("log_likelihood", 0.0)),
            "num_params": int(metrics.get("num_params", 0)),
            "aic": float(metrics.get("aic", 0.0)),
            "bic": float(metrics.get("bic", 0.0)),
            "converged": bool(metrics.get("converged", True)),
            "iterations": int(metrics.get("iterations", 0)),
        },
        "model_parameters": {
            "startprob": model.startprob_.tolist(),
            "transmat": model.transmat_.tolist(),
            "means": model.means_.tolist(),
            "covars": model.covars_.tolist(),
        }
    }

    # 3. Save metadata JSON
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved model metadata to '{metadata_path}'.")

    # 4. Save model selection history CSV
    if summary_df is not None:
        csv_path = os.path.join(os.path.dirname(model_path), "model_selection_history.csv")
        try:
            summary_df.to_csv(csv_path, index=False)
            logger.info(f"Saved model selection history table to '{csv_path}'.")
        except Exception as e:
            logger.warning(f"Could not save model selection history CSV: {e}")


def load_model_and_metadata(
    model_path: str = config.MODEL_FILE,
    metadata_path: str = config.METADATA_FILE,
) -> Tuple[GaussianHMM, Dict[str, Any]]:
    """
    Loads saved HMM model binary and metadata JSON.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    logger.info(f"Successfully loaded HMM model from '{model_path}' and metadata from '{metadata_path}'.")
    return model, metadata
