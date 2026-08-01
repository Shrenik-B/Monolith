from __future__ import annotations

import os
import sys
from pathlib import Path

# Bootstrap project root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
init.py

Purpose:
--------
Generate initial parameters for a Gaussian Hidden Markov Model (HMM)
using K-Means++ clustering on historical market observations.

Outputs:
    - means_init      : Initial cluster centroids (n_states, n_features)
    - covars_init     : Initial emission covariance matrices (n_states, n_features, n_features)
    - startprob_init  : Initial state probability vector (n_states,)
    - labels          : Cluster assignment for each sample (n_samples,)
"""

import logging
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class KMeansInitializer:
    """
    K-Means++ initializer for Gaussian HMM.

    Parameters
    ----------
    n_states : int
        Number of hidden states (market regimes).
    covariance_type : str
        Type of covariance matrix: 'full' or 'diag'. Default is 'full'.
    random_state : int
        Random seed for reproducibility.
    n_init : int
        Number of time the K-Means algorithm will be run with different centroid seeds.
    reg_covar : float
        Regularization term added to the diagonal of covariance matrices to avoid singularity.
    """

    def __init__(
        self,
        n_states: int = 4,
        covariance_type: str = "full",
        random_state: int = 42,
        n_init: int = 20,
        reg_covar: float = 1e-6,
    ):
        if n_states < 2:
            raise ValueError("Number of states (n_states) must be >= 2.")
        if covariance_type not in ("full", "diag"):
            raise ValueError("covariance_type must be either 'full' or 'diag'.")

        self.n_states = n_states
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_init = n_init
        self.reg_covar = reg_covar

    def initialize(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run KMeans++ and estimate initial Gaussian HMM parameters.

        Parameters
        ----------
        X : np.ndarray
            Shape = (n_samples, n_features)

        Returns
        -------
        means_init : np.ndarray
            Shape (n_states, n_features)
        covars_init : np.ndarray
            Shape (n_states, n_features, n_features) for 'full' or (n_states, n_features) for 'diag'
        startprob_init : np.ndarray
            Shape (n_states,)
        labels : np.ndarray
            Cluster assignment for each sample of shape (n_samples,)
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional (got shape {X.shape}).")

        n_samples, n_features = X.shape

        if n_samples == 0:
            raise ValueError("Dataset X is empty.")

        if self.n_states >= n_samples:
            raise ValueError(
                f"Number of states ({self.n_states}) must be smaller than number of samples ({n_samples})."
            )

        logger.info("Running K-Means++ Initialization for Gaussian HMM...")
        logger.info(f"Samples : {n_samples}")
        logger.info(f"Features: {n_features}")
        logger.info(f"States  : {self.n_states}")

        # Step 1: Perform K-Means++ clustering
        kmeans = KMeans(
            n_clusters=self.n_states,
            init="k-means++",
            n_init=self.n_init,
            random_state=self.random_state,
        )

        labels = kmeans.fit_predict(X)
        means_init = kmeans.cluster_centers_

        covars_init = []
        counts = np.zeros(self.n_states, dtype=np.float64)

        # Step 2: Calculate initial covariance matrices & start probabilities per state
        for state in range(self.n_states):
            cluster_data = X[labels == state]
            counts[state] = len(cluster_data)

            if len(cluster_data) <= 1:
                if self.covariance_type == "full":
                    cov = np.eye(n_features, dtype=np.float64)
                else:
                    cov = np.ones(n_features, dtype=np.float64)
            else:
                if self.covariance_type == "full":
                    cov = np.cov(cluster_data, rowvar=False)
                    # Add regularization to guarantee positive-definiteness
                    cov += np.eye(n_features, dtype=np.float64) * self.reg_covar
                else:
                    cov = np.var(cluster_data, axis=0, ddof=1)
                    cov += self.reg_covar

            covars_init.append(cov)

        # Normalize state counts to create initial state probability vector
        startprob_init = counts / counts.sum()
        covars_init = np.asarray(covars_init)

        logger.info("K-Means++ Initialization Complete.")
        logger.info(f"Inertia : {kmeans.inertia_:.4f}")

        return (
            means_init,
            covars_init,
            startprob_init,
            labels,
        )
