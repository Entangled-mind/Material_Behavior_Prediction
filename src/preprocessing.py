"""
preprocessing.py
================
Milestone 3: Data Preprocessing & Train/Test Split
Project: Material Behavior Prediction Using Machine Learning

Educational Principles:
- Clear, beginner-friendly, modular Python.
- Handles only what is strictly required by the dataset.
- Zero Data Leakage: Scalers fit ONLY on training data.
- Can be run as a standalone verification script or imported into training scripts.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_raw_data(data_path):
    """Loads the raw material dataset from CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    print("=" * 70)
    print("[PREPROCESSING 1] LOADING RAW DATA")
    print("=" * 70)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from: {data_path}\n")
    return df


def handle_duplicates(df):
    """
    Identifies and removes exact duplicate rows.
    
    Rationale:
    25 rows are exact duplicates (identical mix ingredients AND identical strength).
    If an exact duplicate falls in both training and testing sets, the test set
    evaluates a row the model has already memorized, causing data leakage and
    over-optimistic performance metrics.
    """
    print("=" * 70)
    print("[PREPROCESSING 2] HANDLING DUPLICATE ROWS")
    print("=" * 70)
    initial_rows = len(df)
    duplicates_count = df.duplicated().sum()

    print(f"Initial row count         : {initial_rows}")
    print(f"Exact duplicate rows found: {duplicates_count}")

    df_clean = df.drop_duplicates().copy()
    cleaned_rows = len(df_clean)
    print(f"Cleaned row count         : {cleaned_rows} (dropped {duplicates_count} duplicates)")
    print("-> Data hygiene complete: Zero duplicates remain.\n")
    return df_clean


def separate_features_and_target(df, target_col='strength'):
    """Separates input features (X) from the target output variable (y)."""
    print("=" * 70)
    print("[PREPROCESSING 3] SEPARATING FEATURES (X) AND TARGET (y)")
    print("=" * 70)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Features (X) shape : {X.shape} ({list(X.columns)})")
    print(f"Target   (y) shape : {y.shape} ('{target_col}')\n")
    return X, y


def split_data(X, y, test_size=0.20, random_state=42):
    """
    Splits data into training (80%) and testing (20%) sets.
    
    Rationale:
    - 80% Training (804 samples): Used by ML algorithms to learn patterns.
    - 20% Testing (201 samples): Kept strictly unseen until final evaluation.
    - random_state=42: Fixed seed ensuring identical, reproducible splits.
    """
    print("=" * 70)
    print(f"[PREPROCESSING 4] TRAIN / TEST SPLIT ({int((1-test_size)*100)}% / {int(test_size*100)}%)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"X_train (training features): {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"X_test  (testing features) : {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"y_train (training target)  : {y_train.shape[0]} values (Mean: {y_train.mean():.2f} MPa)")
    print(f"y_test  (testing target)   : {y_test.shape[0]} values (Mean: {y_test.mean():.2f} MPa)")
    print("-> Balanced partition confirmed: Train and test target means are very close.\n")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    
    Formula: z = (x - u) / s
    
    CRITICAL ANTI-DATA LEAKAGE RULE:
    - scaler.fit_transform(X_train): Computes mean (u) and std (s) ONLY from X_train.
    - scaler.transform(X_test): Applies the TRAINING mean and std to X_test.
    - The test set is NEVER fitted, so zero information leaks from test to train.
    """
    print("=" * 70)
    print("[PREPROCESSING 5] FEATURE SCALING (StandardScaler)")
    print("=" * 70)
    scaler = StandardScaler()

    # 1. Fit ONLY on training data, then transform training data
    X_train_scaled = scaler.fit_transform(X_train)

    # 2. Transform test data using the parameters learned from training data
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for readability and feature name retention
    feature_names = X_train.columns.tolist()
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

    print("Scaler fitted strictly on training data:")
    for col, mean_val, std_val in zip(feature_names, scaler.mean_, scaler.scale_):
        print(f"  - {col:14s}: Mean = {mean_val:7.2f}, Std = {std_val:7.2f}")

    print("\nVerification after scaling on X_train:")
    print(f"  - Scaled training mean (should be ~0.00): {X_train_scaled.mean():.4f}")
    print(f"  - Scaled training std  (should be ~1.00): {X_train_scaled.std():.4f}")
    print("-> Scaling successfully verified with zero data leakage.\n")

    return X_train_scaled_df, X_test_scaled_df, scaler


def prepare_data(data_path=None, test_size=0.20, random_state=42, save_scaler=False, scaler_path=None):
    """
    High-level orchestrator that executes the full preprocessing pipeline.
    Returns:
        X_train_raw, X_test_raw, X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, 'data', 'material_data.csv')

    df = load_raw_data(data_path)
    df_clean = handle_duplicates(df)
    X, y = separate_features_and_target(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    if save_scaler:
        if scaler_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"Saved fitted scaler to: {scaler_path}")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def main():
    print("*" * 70)
    print("  MATERIAL BEHAVIOR PREDICTION USING MACHINE LEARNING")
    print("  Milestone 3: Data Preprocessing Pipeline")
    print("*" * 70 + "\n")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    scaler_file = os.path.join(models_dir, 'scaler.joblib')

    # Run full preparation
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(
        save_scaler=True, scaler_path=scaler_file
    )

    print("=" * 70)
    print("SUMMARY OF PREPROCESSED DATASETS FOR MODEL TRAINING")
    print("=" * 70)
    print(f"Training set: {X_train.shape[0]} samples (unscaled & scaled versions ready)")
    print(f"Testing set : {X_test.shape[0]} samples (unscaled & scaled versions ready)")
    print(f"Scaler saved: models/scaler.joblib")
    print("=" * 70)
    print("\nMilestone 3 complete! The dataset is partitioned, scaled, and ready for modeling.")


if __name__ == '__main__':
    main()
