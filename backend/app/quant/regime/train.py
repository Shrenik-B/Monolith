from __future__ import annotations

"""
train.py

Purpose:
--------
1. Initializes Gaussian HMM using K-Means++ parameters.
2. Trains the Gaussian HMM using the Baum-Welch (EM) algorithm.
3. Evaluates candidate models across different hidden state numbers (e.g. 3, 4, 5, 6, 7).
4. Computes Log-Likelihood, AIC, and BIC metrics.
5. Selects the optimal model based on minimum BIC.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from backend.app.quant.regime.load import retrieve_features
from backend.app.quant.regime.init import KMeansInitializer
from backend.app.core import config

logger = logging.getLogger(__name__)


def compute_num_parameters(n_states: int, n_features: int, covariance_type: str = "full") -> int:
    """
    Computes total number of free parameters K for a Gaussian HMM.
    """
    # Initial state probabilities: (n_states - 1)
    startprob_params = n_states - 1
    # Transition probability matrix: n_states * (n_states - 1)
    transmat_params = n_states * (n_states - 1)
    # Emission means: n_states * n_features
    means_params = n_states * n_features

    # Emission covariances
    if covariance_type == "full":
        cov_params = n_states * (n_features * (n_features + 1) // 2)
    elif covariance_type == "diag":
        cov_params = n_states * n_features
    else:
        raise ValueError(f"Unsupported covariance type: {covariance_type}")

    total_k = startprob_params + transmat_params + means_params + cov_params
    return total_k


def train_single_hmm(
    X: np.ndarray,
    n_states: int,
    covariance_type: str = "full",
    random_state: int = 42,
    n_init: int = 20,
    n_iter: int = 100,
    tol: float = 1e-4,
    reg_covar: float = 1e-6,
) -> Tuple[GaussianHMM, Dict[str, Any]]:
    """
    Trains a single Gaussian HMM using K-Means++ initialization and Baum-Welch (EM).
    """
    n_samples, n_features = X.shape

    # 1. Initialize parameters using K-Means++
    initializer = KMeansInitializer(
        n_states=n_states,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
        reg_covar=reg_covar,
    )
    means_init, covars_init, startprob_init, labels = initializer.initialize(X)

    # 2. Instantiate Gaussian HMM with pre-computed initial values
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        init_params="",  # Do not randomly initialize parameters; use K-Means++ values
        params="stmc",   # Refine startprob, transmat, means, covars during Baum-Welch
        min_covar=reg_covar,
    )

    model.startprob_ = startprob_init
    model.means_ = means_init
    model.covars_ = covars_init
    
    # Initialize transition matrix uniformly
    model.transmat_ = np.full((n_states, n_states), 1.0 / n_states)

    # 3. Fit model using Baum-Welch (EM) algorithm
    model.fit(X)

    # 4. Evaluate Log-Likelihood, AIC, BIC
    log_likelihood = float(model.score(X))
    num_params = compute_num_parameters(n_states, n_features, covariance_type)
    
    aic = 2 * num_params - 2 * log_likelihood
    bic = num_params * np.log(n_samples) - 2 * log_likelihood

    metrics = {
        "n_states": n_states,
        "n_samples": n_samples,
        "n_features": n_features,
        "log_likelihood": log_likelihood,
        "num_params": num_params,
        "aic": aic,
        "bic": bic,
        "converged": model.monitor_.converged,
        "iterations": model.monitor_.iter,
    }

    return model, metrics


def select_best_hmm(
    X: np.ndarray,
    candidate_states: List[int] = [3, 4, 5, 6, 7],
    covariance_type: str = "full",
    random_state: int = 42,
) -> Tuple[GaussianHMM, Dict[str, Any], pd.DataFrame]:
    """
    Trains HMMs for all candidate states and selects the best model using minimum BIC.
    """
    logger.info(f"Starting Model Selection across candidate states: {candidate_states}")
    
    results_list = []
    trained_models = {}

    for n_states in candidate_states:
        print(f"Training Gaussian HMM with K={n_states} states...")
        model, metrics = train_single_hmm(
            X=X,
            n_states=n_states,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=config.N_INIT,
            n_iter=config.N_ITER,
            tol=config.TOL,
            reg_covar=config.REG_COVAR,
        )
        results_list.append(metrics)
        trained_models[n_states] = model

    summary_df = pd.DataFrame(results_list)
    summary_df = summary_df.sort_values(by="bic", ascending=True).reset_index(drop=True)

    best_n_states = int(summary_df.loc[0, "n_states"])
    best_model = trained_models[best_n_states]
    best_metrics = summary_df.loc[0].to_dict()

    print("\n=======================================================")
    print("           HMM MODEL SELECTION SUMMARY                 ")
    print("=======================================================")
    print(summary_df[["n_states", "num_params", "log_likelihood", "aic", "bic", "converged"]].to_string(index=False))
    print("-------------------------------------------------------")
    print(f" Best Model Selected: K = {best_n_states} hidden states (Lowest BIC = {best_metrics['bic']:.2f})")
    print("=======================================================\n")

    return best_model, best_metrics, summary_df


if __name__ == "__main__":
    X, feature_names, df_clean = retrieve_features(config.PARQUET_FILE, use_only_z_features=config.USE_ONLY_Z_FEATURES)
    best_model, best_metrics, summary_df = select_best_hmm(X, candidate_states=config.CANDIDATE_STATES)
