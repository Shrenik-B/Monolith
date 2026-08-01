import pandas as pd
import numpy as np

# Set display options for clean terminal output
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

def retrieve_features(file_name="features_v1.parquet", use_only_z_features=False):
    """
    Retrieves the complete dataset (X feature matrix) ready for model training.
    
    Parameters:
        file_name (str): Path to parquet file.
        use_only_z_features (bool): If True, loads only '_z' normalized columns. 
                                    If False (default), loads all numeric feature columns.

    Returns:
        X (np.ndarray): Full feature matrix
        feature_cols (list): Names of feature columns used
        df_clean (pd.DataFrame): Cleaned source Pandas DataFrame
    """
    print(f"Loading full dataset from '{file_name}'...")

    try:
        df = pd.read_parquet(file_name)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return None, None, None

    total_columns_in_parquet = list(df.columns)

    # 1. Filter valid training rows if flag exists
    if 'is_training_valid' in df.columns:
        df = df[df['is_training_valid'] == True]

    # 2. Determine feature columns
    if use_only_z_features:
        feature_cols = [col for col in df.columns if col.endswith('_z')]
    else:
        # Explicitly exclude metadata columns that would crash the math model
        metadata_cols = ['date', 'era_tag', 'feature_version', 'vintage_date', 'is_training_valid']
        
        # Keep column if it is NOT metadata AND it contains numeric data
        feature_cols = [
            col for col in df.columns 
            if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

    # 3. Ensure required columns are clean (no NaNs)
    df_clean = df.dropna(subset=feature_cols)

    # 4. Extract X (features matrix) and standardize (z-score) across columns for balanced model training
    raw_X = df_clean[feature_cols].values.astype(np.float64)
    means = np.mean(raw_X, axis=0)
    stds = np.std(raw_X, axis=0)
    stds[stds == 0] = 1.0
    X = (raw_X - means) / stds

    # 5. Terminal Display
    print("\n--- Parquet Dataset Inspection ---")
    print(f" Total Parquet Columns ({len(total_columns_in_parquet)}): {', '.join(total_columns_in_parquet)}")
    print("\n--- Extracted Data Summary ---")
    print(f" Total Rows (n_samples):   {X.shape[0]}")
    print(f" Features Matrix (X):      shape={X.shape} ({len(feature_cols)} features)")
    print(f" Features Included in X:   \n   - " + "\n   - ".join(feature_cols))
    print("-----------------------------------\n")

    return X, feature_cols, df_clean


if __name__ == "__main__":
    # If you only want '_z' columns, change to True. 
    # If you want all 16 numeric features, keep as False.
    X, feature_names, df = retrieve_features(use_only_z_features=False)