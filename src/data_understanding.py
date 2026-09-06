"""
data_understanding.py
=====================
Milestone 1: Data Understanding
Project: Material Behavior Prediction Using Machine Learning

Educational Principles:
- Beginner-friendly, highly readable Python.
- Transparent inspection of features, target, data types, and hygiene.
- No silent assumptions or premature preprocessing.
"""

import os
import pandas as pd
import numpy as np

# Configure pandas formatting for clear console output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def load_dataset(file_path):
    """
    Loads the dataset from the specified file path.
    Includes explicit error handling with helpful guidance if missing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: Dataset not found at '{file_path}'.\n"
            "Please verify that 'material_data.csv' is placed inside the 'data/' folder."
        )
    print("=" * 70)
    print(f"[STEP 1] LOADING DATASET")
    print("=" * 70)
    print(f"File path: {file_path}")
    df = pd.read_csv(file_path)
    print("-> Successfully loaded dataset into pandas DataFrame.\n")
    return df


def inspect_dataset(df):
    """
    Executes Milestone 1 Data Understanding checks sequentially:
    1. First and last rows
    2. Shape (rows x columns)
    3. Column names
    4. Data types and non-null counts (info)
    5. Statistical summary (describe)
    6. Missing values audit
    7. Duplicate rows check
    8. Numerical vs categorical column breakdown
    9. Physical / problematic values check
    10. Target variable & scientific definition
    """

    # ------------------------------------------------------------------
    # Step 2: Display First and Last Rows
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[STEP 2] FIRST 5 ROWS (head)")
    print("=" * 70)
    print(df.head())
    print("\n")

    print("=" * 70)
    print("[STEP 2.1] LAST 5 ROWS (tail)")
    print("=" * 70)
    print(df.tail())
    print("\n")

    # ------------------------------------------------------------------
    # Step 3: Dataset Shape
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[STEP 3] DATASET SHAPE")
    print("=" * 70)
    num_rows, num_cols = df.shape
    print(f"Total rows (samples/experiments) : {num_rows}")
    print(f"Total columns (features + target): {num_cols}")
    print("\n")

    # ------------------------------------------------------------------
    # Step 4: Column Names
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[STEP 4] COLUMN NAMES")
    print("=" * 70)
    for idx, col in enumerate(df.columns, start=1):
        print(f"  {idx}. {col}")
    print("\n")

    # ------------------------------------------------------------------
    # Step 5: DataFrame Info (Types and Memory)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[STEP 5] DATA TYPES AND NON-NULL COUNTS (info)")
    print("=" * 70)
    df.info()
    print("\n")

    # ------------------------------------------------------------------
    # Step 6: Statistical Summary (describe)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[STEP 6] STATISTICAL SUMMARY (describe)")
    print("=" * 70)
    summary_stats = df.describe().round(2)
    print(summary_stats)
    print("\n")

    # ------------------------------------------------------------------
    # Step 7: Missing Values
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[STEP 7] MISSING VALUES AUDIT")
    print("=" * 70)
    missing_series = df.isnull().sum()
    total_missing = missing_series.sum()
    print(f"Total missing values across entire dataset: {total_missing}")
    if total_missing == 0:
        print("-> Clean: No missing values detected in any column.")
    else:
        print("Missing values per column:")
        print(missing_series[missing_series > 0])
    print("\n")

    # ------------------------------------------------------------------
    # Step 8: Duplicate Rows
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[STEP 8] DUPLICATE ROWS CHECK")
    print("=" * 70)
    num_duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {num_duplicates} (out of {num_rows} rows)")
    if num_duplicates > 0:
        pct_dup = (num_duplicates / num_rows) * 100
        print(f"Percentage of duplicate rows: {pct_dup:.2f}%")
        print("\nDomain Explanation:")
        print("In material testing experiments, researchers often prepare multiple")
        print("identical batches or test duplicate specimens to measure variability.")
        print("During Milestone 3 (Preprocessing), we will examine whether these are")
        print("exact identical replicates or repeated experimental measurements.")
    else:
        print("-> No duplicate rows detected.")
    print("\n")

    # ------------------------------------------------------------------
    # Step 9: Column Categorization (Numerical vs Categorical)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[STEP 9] NUMERICAL AND CATEGORICAL FEATURE IDENTIFICATION")
    print("=" * 70)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numerical Columns ({len(numerical_cols)}):")
    for col in numerical_cols:
        print(f"  - {col} (dtype: {df[col].dtype})")

    print(f"\nCategorical Columns ({len(categorical_cols)}):")
    if categorical_cols:
        for col in categorical_cols:
            print(f"  - {col}")
    else:
        print("  - None (All variables in this dataset are quantitative measurements).")
    print("\n")

    # ------------------------------------------------------------------
    # Step 10: Problematic and Physical Boundary Checks
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[STEP 10] PHYSICAL BOUNDARY & VALUE VALIDATION")
    print("=" * 70)
    negative_detected = False
    for col in numerical_cols:
        min_v = df[col].min()
        if min_v < 0:
            print(f"  WARNING: Physical violation! Column '{col}' has negative minimum: {min_v}")
            negative_detected = True
    if not negative_detected:
        print("-> Physical boundary check passed: All numerical values >= 0 (no negative masses/times).")

    print("\nSparsity / Zero-value check (optional mix ingredients):")
    zero_counts = (df == 0).sum()
    for col, z_count in zero_counts.items():
        if z_count > 0:
            z_pct = (z_count / num_rows) * 100
            print(f"  - {col:15s}: {z_count:4d} zero values ({z_pct:5.1f}% of batches)")
    print("\n")

    # ------------------------------------------------------------------
    # Step 11: Target Variable & Scientific Problem Definition
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[STEP 11] TARGET IDENTIFICATION & SCIENTIFIC PROBLEM FORMULATION")
    print("=" * 70)
    target_col = 'strength'
    features = [c for c in df.columns if c != target_col]

    print(f"Identified Target Variable (y): '{target_col}'")
    print("  * What it represents : Compressive Strength of concrete (in MPa - Megapascals)")
    print("  * ML Problem Type    : Supervised Regression (continuous numerical target)")
    print(f"  * Minimum Strength   : {df[target_col].min():.2f} MPa")
    print(f"  * Maximum Strength   : {df[target_col].max():.2f} MPa")
    print(f"  * Mean Strength      : {df[target_col].mean():.2f} MPa")
    print(f"  * Median Strength    : {df[target_col].median():.2f} MPa")
    print(f"  * Standard Deviation : {df[target_col].std():.2f} MPa")

    print(f"\nInput Features (X) ({len(features)} features):")
    feature_meta = {
        'cement': 'Portland cement content (kg in a m^3 mixture)',
        'slag': 'Blast furnace slag (kg in a m^3 mixture) - supplementary cementitious material',
        'ash': 'Fly ash (kg in a m^3 mixture) - supplementary cementitious material',
        'water': 'Water content (kg in a m^3 mixture)',
        'superplastic': 'Superplasticizer (kg in a m^3 mixture) - chemical workability additive',
        'coarseagg': 'Coarse aggregate/gravel (kg in a m^3 mixture)',
        'fineagg': 'Fine aggregate/sand (kg in a m^3 mixture)',
        'age': 'Curing age prior to testing (days, e.g. 1 to 365 days)'
    }
    for f in features:
        desc = feature_meta.get(f, 'Experimental variable')
        print(f"  - {f:14s}: {desc}")

    print("\nScientific Problem Mapping:")
    print("  [Material Composition + Processing Age]  ---> [Mechanical Behavior (Compressive Strength)]")
    print("=" * 70)


def main():
    # Resolve file path dynamically relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_file = os.path.join(project_root, "data", "material_data.csv")

    print("*" * 70)
    print("  MATERIAL BEHAVIOR PREDICTION USING MACHINE LEARNING")
    print("  Milestone 1: Data Understanding Pipeline")
    print("*" * 70 + "\n")

    df = load_dataset(data_file)
    inspect_dataset(df)

    print("\nMilestone 1 complete. Dataset is understood and ready for EDA (Milestone 2).")


if __name__ == "__main__":
    main()
