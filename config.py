"""
config.py

Configuration parameters for K-Means++ HMM Initialization, Training, Model Selection, Persistence, Inference, and DTW Engine.
"""

# File paths
PARQUET_FILE = "features_v1.parquet"
MODEL_FILE = "model.pkl"
METADATA_FILE = "model_metadata.json"

# Feature Selection
USE_ONLY_Z_FEATURES = False    # False = all 17 features, True = 10 z-scored features

# HMM / K-Means Initializer Configuration
N_STATES = 4                  # Default number of hidden market regimes
RANDOM_STATE = 42             # Seed for reproducible initialization
N_INIT = 60                   # Number of K-Means runs with different seeds (higher = better initialization)
REG_COVAR = 1e-6              # Regularization added to covariance matrices
COVARIANCE_TYPE = "full"      # 'full' or 'diag'
N_ITER = 200                  # Maximum Baum-Welch (EM) iterations (higher = tighter convergence)
TOL = 1e-4                    # EM convergence tolerance

# Model Selection Candidates
CANDIDATE_STATES = [3, 4, 5, 6, 7]  # Number of hidden states to evaluate for AIC/BIC selection (K=3 to K=7)

# DTW Engine Configuration
DTW_WINDOW_SIZE = 30          # Number of trading days in trajectory window
DTW_TOP_N = 3                 # Number of most similar historical matches to return
DTW_STEP_SIZE = 5             # Rolling step size for historical search
