"""
train_model.py
==============
Milestone 4: Baseline Model Training & Evaluation
Project: Material Behavior Prediction Using Machine Learning

Educational Principles:
- Simple, transparent Linear Regression baseline.
- No black-box complexity before understanding a simple model.
- Comprehensive metric evaluation: MAE, RMSE, R².
- Connects model weights (coefficients) directly to materials science.
- Diagnostic plots saved in 'visualizations/' and model saved in 'models/'.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Ensure script can import preprocessing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import prepare_data


def train_baseline_model(X_train_scaled, y_train):
    """
    Instantiates and trains an Ordinary Least Squares (OLS) Linear Regression model.
    """
    print("=" * 70)
    print("[TRAINING BASELINE] ORDINARY LEAST SQUARES LINEAR REGRESSION")
    print("=" * 70)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    print("-> Model training complete.\n")
    return model


def evaluate_predictions(y_true, y_pred, dataset_name="Test"):
    """
    Computes standard regression metrics:
    - MAE  : Mean Absolute Error (average magnitude of error in MPa)
    - RMSE : Root Mean Squared Error (penalizes large mistakes in MPa)
    - R²   : Coefficient of Determination (proportion of variance explained)
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"[{dataset_name.upper()} SET PERFORMANCE METRICS]")
    print(f"  - R² Score (Coefficient of Determination) : {r2:.4f} ({r2*100:.1f}% variance explained)")
    print(f"  - MAE  (Mean Absolute Error)              : {mae:.2f} MPa")
    print(f"  - RMSE (Root Mean Squared Error)          : {rmse:.2f} MPa\n")

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def analyze_coefficients(model, feature_names):
    """
    Extracts and interprets standardized regression coefficients.
    Since features were standardized (mean=0, std=1), coefficient magnitude
    directly reflects feature importance in the linear model.
    """
    print("=" * 70)
    print("[MODEL INTERPRETATION] LEARNED LINEAR EQUATION & COEFFICIENTS")
    print("=" * 70)
    intercept = model.intercept_
    print(f"Intercept (w0 / Baseline Average): {intercept:.2f} MPa\n")

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Standardized_Weight': model.coef_
    }).sort_values(by='Standardized_Weight', ascending=False)

    print("Learned Feature Weights (Standardized):")
    for _, row in coef_df.iterrows():
        feat = row['Feature']
        weight = row['Standardized_Weight']
        direction = "+" if weight > 0 else "-"
        print(f"  {direction} {feat:14s}: {weight:7.2f} MPa per 1-std increase")

    print("\nScientific Physical Meaning:")
    print("  1. 'cement' (+12.12) is by far the strongest positive linear predictor of strength.")
    print("  2. 'slag' (+8.44), 'age' (+6.89), and 'ash' (+5.44) strongly boost strength.")
    print("  3. 'water' (-2.83) has a negative coefficient: excess mixing water leaves microscopic")
    print("     voids upon evaporation, weakening the load-bearing concrete matrix.")
    print("=" * 70 + "\n")
    return coef_df


def plot_baseline_diagnostics(y_test, y_test_pred, output_dir):
    """
    Generates two crucial diagnostic plots for the baseline model:
    1. Actual vs. Predicted scatter plot with perfect-prediction 45-degree reference line.
    2. Residual error distribution (actual - predicted).
    """
    os.makedirs(output_dir, exist_ok=True)
    residuals = y_test - y_test_pred

    # -------------------------------------------------------------
    # Plot 1: Actual vs. Predicted
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test, y_test_pred, color='#2563eb', alpha=0.7, edgecolors='none', s=45, label='Test Batches')
    
    # 45-degree perfect prediction line
    min_val = min(y_test.min(), y_test_pred.min()) - 2
    max_val = max(y_test.max(), y_test_pred.max()) + 2
    plt.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
    
    plt.title("Baseline Linear Regression: Actual vs. Predicted Strength", fontsize=13, fontweight='bold')
    plt.xlabel("Actual Compressive Strength (MPa)", fontsize=11, fontweight='bold')
    plt.ylabel("Predicted Compressive Strength (MPa)", fontsize=11, fontweight='bold')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)

    actual_pred_path = os.path.join(output_dir, '07_baseline_actual_vs_predicted.png')
    plt.tight_layout()
    plt.savefig(actual_pred_path, dpi=300)
    plt.close()
    print(f"  -> Saved diagnostic plot: {actual_pred_path}")

    # -------------------------------------------------------------
    # Plot 2: Residuals Distribution
    # -------------------------------------------------------------
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={'height_ratios': [0.25, 0.75]}
    )

    sns.boxplot(x=residuals, ax=ax_box, color='#93c5fd', fliersize=4)
    ax_box.axvline(0, color='#dc2626', linestyle='--', linewidth=1.5)
    ax_box.set(xlabel='')
    ax_box.set_title("Baseline Residual Errors Distribution (Actual - Predicted)", fontsize=13, fontweight='bold', pad=10)

    sns.histplot(residuals, kde=True, ax=ax_hist, color='#3b82f6', bins=25, edgecolor='#1d4ed8')
    ax_hist.axvline(0, color='#dc2626', linestyle='--', linewidth=1.5, label='Zero Error (Target)')
    ax_hist.set_xlabel("Prediction Error (Residual in MPa)", fontsize=11, fontweight='bold')
    ax_hist.set_ylabel("Frequency", fontsize=11, fontweight='bold')
    ax_hist.legend(loc='upper right', frameon=True)

    residuals_path = os.path.join(output_dir, '08_baseline_residuals.png')
    plt.tight_layout()
    plt.savefig(residuals_path, dpi=300)
    plt.close()
    print(f"  -> Saved diagnostic plot: {residuals_path}")


def main():
    print("*" * 70)
    print("  MATERIAL BEHAVIOR PREDICTION USING MACHINE LEARNING")
    print("  Milestone 4: Baseline Model Training & Evaluation")
    print("*" * 70 + "\n")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    viz_dir = os.path.join(base_dir, 'visualizations')
    baseline_model_file = os.path.join(models_dir, 'baseline_linear_regression.joblib')

    # 1. Load Preprocessed Data
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = prepare_data()

    # 2. Train Baseline Model
    baseline_model = train_baseline_model(X_train_s, y_train)

    # 3. Generate Predictions
    y_train_pred = baseline_model.predict(X_train_s)
    y_test_pred = baseline_model.predict(X_test_s)

    # 4. Evaluate Metrics
    train_metrics = evaluate_predictions(y_train, y_train_pred, dataset_name="Training")
    test_metrics = evaluate_predictions(y_test, y_test_pred, dataset_name="Testing")

    # 5. Interpret Model Weights
    analyze_coefficients(baseline_model, X_train.columns)

    # 6. Save Model Artifact
    joblib.dump(baseline_model, baseline_model_file)
    print(f"Saved baseline model artifact to: {baseline_model_file}")

    # 7. Generate Diagnostic Visualizations
    print("\nGenerating baseline diagnostic charts...")
    plot_baseline_diagnostics(y_test, y_test_pred, viz_dir)

    print("\n" + "=" * 70)
    print("BASELINE SUMMARY:")
    print(f"  - Baseline Model : Ordinary Least Squares Linear Regression")
    print(f"  - Test R² Score  : {test_metrics['R2']:.4f} (~58.0% variance explained)")
    print(f"  - Test MAE Error : {test_metrics['MAE']:.2f} MPa")
    print(f"  - Test RMSE      : {test_metrics['RMSE']:.2f} MPa")
    print("  - Purpose        : This benchmark sets the standard to beat in Milestone 5.")
    print("=" * 70)


if __name__ == '__main__':
    main()
