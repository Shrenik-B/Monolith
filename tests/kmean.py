"""
tests/kmean.py

Purpose:
--------
Loads macroeconomic market features from Parquet and runs standalone K-Means++ initialization 
to test parameter generation (means_init, covars_init, startprob_init).
"""

from __future__ import annotations

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import numpy as np
import pandas as pd

from load import retrieve_features
from init import KMeansInitializer
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_kmeans_initialization():
    X, feature_names, df = retrieve_features(file_name=config.PARQUET_FILE, use_only_z_features=False)
    if X is None:
        print("Error: Failed to load features matrix X.")
        return None

    initializer = KMeansInitializer(
        n_states=config.N_STATES,
        covariance_type=config.COVARIANCE_TYPE,
        random_state=config.RANDOM_STATE,
        n_init=config.N_INIT,
        reg_covar=config.REG_COVAR,
    )

    means_init, covars_init, startprob_init, labels = initializer.initialize(X)

    print("\n=======================================================")
    print("      K-MEANS++ HMM INITIALIZATION TEST RESULTS        ")
    print("=======================================================")
    print(f" Feature Matrix X Shape : {X.shape} ({X.shape[0]} samples, {X.shape[1]} features)")
    print(f" Number of HMM States (K): {config.N_STATES}")
    print(f" Initial Means Shape    : {means_init.shape}")
    print(f" Initial Covars Shape   : {covars_init.shape}")
    print("=======================================================\n")

    return {
        "means_init": means_init,
        "covars_init": covars_init,
        "startprob_init": startprob_init,
        "labels": labels,
    }

if __name__ == "__main__":
    run_kmeans_initialization()
