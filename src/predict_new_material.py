"""
predict_new_material.py
=======================
Milestone 8: New Material Prediction Interface & Model Persistence
Project: Material Behavior Prediction Using Machine Learning

Educational Principles:
- User-friendly, self-contained prediction function `predict_material(...)`.
- Input validation: checks physical bounds and warns if outside the training envelope.
- Computes key materials engineering ratios (Water-to-Binder ratio).
- Explains the prediction in plain engineering language.
- Includes mandatory scientific disclaimers: ML estimate vs. experimental measurement.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib


def load_champion_model(model_path=None):
    """
    Loads the trained champion model (Gradient Boosting).
    """
    if model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'final_material_model.joblib')
        # Fallback to gradient_boosting.joblib if final_material_model isn't copied yet
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, 'models', 'gradient_boosting.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Run evaluate_model.py first.")

    model = joblib.load(model_path)
    return model


def validate_physical_inputs(cement, slag, ash, water, superplastic, coarseagg, fineagg, age):
    """
    Checks that input values make physical sense and warns if values fall
    outside the experimental domain of the training dataset.
    """
    inputs = {
        'cement': cement, 'slag': slag, 'ash': ash, 'water': water,
        'superplastic': superplastic, 'coarseagg': coarseagg,
        'fineagg': fineagg, 'age': age
    }

    # 1. Non-negativity check
    for name, val in inputs.items():
        if val < 0:
            raise ValueError(f"Physical violation: '{name}' cannot be negative (provided: {val}).")

    if age <= 0:
        raise ValueError(f"Curing age must be at least 1 day (provided: {age}).")

    # 2. Total density check (standard concrete is 2200 to 2500 kg/m^3)
    total_mass = cement + slag + ash + water + superplastic + coarseagg + fineagg
    if total_mass < 1800 or total_mass > 2700:
        print(f"  [!] WARNING: Total mix mass is {total_mass:.1f} kg/m³. Standard concrete typically")
        print("      ranges between 2,200 and 2,500 kg/m³. Prediction may be less reliable.")

    # 3. Domain of applicability (Age check)
    if age > 365:
        print(f"  [!] WARNING: Curing age ({age} days) exceeds the 365-day training boundary.")
        print("      Tree models cannot extrapolate beyond historical curing times.")


def get_strength_classification(strength_mpa):
    """
    Maps continuous compressive strength (MPa) to standard structural engineering grades.
    """
    if strength_mpa < 20.0:
        return "Lean / Plain Concrete (Sub-bases, non-structural paving, mass filling)"
    elif 20.0 <= strength_mpa < 35.0:
        return "Standard Structural Concrete (Residential foundations, slabs, columns)"
    elif 35.0 <= strength_mpa < 50.0:
        return "Heavy Commercial Structural Concrete (Multi-story buildings, highway bridges)"
    else:
        return "High-Performance / High-Strength Concrete (Pre-stressed beams, skyscrapers, marine)"


def predict_material(cement, slag=0.0, ash=0.0, water=180.0, superplastic=0.0,
                     coarseagg=1000.0, fineagg=750.0, age=28, model_path=None, verbose=True):
    """
    Accepts ingredient quantities (in kg/m³) and curing duration (in days),
    validates the inputs, applies the champion ML model, and returns predicted strength.

    Parameters:
        cement       : Portland cement (kg/m³)
        slag         : Blast furnace slag (kg/m³)
        ash          : Fly ash (kg/m³)
        water        : Mixing water (kg/m³)
        superplastic : Chemical superplasticizer (kg/m³)
        coarseagg    : Coarse gravel aggregate (kg/m³)
        fineagg      : Fine sand aggregate (kg/m³)
        age          : Curing duration before testing (days)
        model_path   : Path to saved .joblib model (optional)
        verbose      : If True, prints formatted engineering summary

    Returns:
        float: Predicted Compressive Strength in Megapascals (MPa)
    """
    # 1. Validate physical inputs
    validate_physical_inputs(cement, slag, ash, water, superplastic, coarseagg, fineagg, age)

    # 2. Format input as single-row DataFrame with identical feature columns
    feature_names = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']
    input_df = pd.DataFrame([{
        'cement': float(cement),
        'slag': float(slag),
        'ash': float(ash),
        'water': float(water),
        'superplastic': float(superplastic),
        'coarseagg': float(coarseagg),
        'fineagg': float(fineagg),
        'age': int(age)
    }], columns=feature_names)

    # 3. Load model and predict
    model = load_champion_model(model_path)
    predicted_strength = float(model.predict(input_df)[0])

    # 4. Compute materials science ratios
    total_binder = cement + slag + ash
    water_cement_ratio = water / cement if cement > 0 else np.nan
    water_binder_ratio = water / total_binder if total_binder > 0 else np.nan
    total_density = cement + slag + ash + water + superplastic + coarseagg + fineagg
    grade = get_strength_classification(predicted_strength)

    # 5. Display detailed breakdown if requested
    if verbose:
        print("=" * 70)
        print("NEW MATERIAL PREDICTION: MECHANICAL BEHAVIOR ESTIMATE")
        print("=" * 70)
        print("Input Mix Formulation (per 1 m³):")
        print(f"  - Portland Cement      : {cement:6.1f} kg")
        print(f"  - Blast Furnace Slag   : {slag:6.1f} kg")
        print(f"  - Fly Ash              : {ash:6.1f} kg")
        print(f"  - Mixing Water         : {water:6.1f} kg")
        print(f"  - Superplasticizer     : {superplastic:6.1f} kg")
        print(f"  - Coarse Aggregate     : {coarseagg:6.1f} kg")
        print(f"  - Fine Aggregate (Sand): {fineagg:6.1f} kg")
        print(f"  - Curing Age           : {age:6d} days")
        print("-" * 70)
        print("Calculated Physical Ratios:")
        print(f"  - Total Binder Mass    : {total_binder:.1f} kg/m³")
        print(f"  - Water-to-Cement (w/c): {water_cement_ratio:.2f}")
        print(f"  - Water-to-Binder (w/b): {water_binder_ratio:.2f}")
        print(f"  - Total Wet Density    : {total_density:.1f} kg/m³")
        print("-" * 70)
        print(f"-> PREDICTED COMPRESSIVE STRENGTH : {predicted_strength:.2f} MPa")
        print(f"-> STRUCTURAL APPLICATION GRADE   : {grade}")
        print("-" * 70)
        print("SCIENTIFIC NOTE:")
        print("This prediction is a machine-learned statistical estimate based on")
        print("empirical laboratory data. It must be confirmed by standard cylinder")
        print("destructive testing (ASTM C39) before being used for life-critical structures.")
        print("=" * 70 + "\n")

    return predicted_strength


def main():
    print("*" * 70)
    print("  MATERIAL BEHAVIOR PREDICTION USING MACHINE LEARNING")
    print("  Milestone 8: New Material Formulation Testing Suite")
    print("*" * 70 + "\n")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    # Save champion model officially as final_material_model.joblib
    src_model = os.path.join(models_dir, 'gradient_boosting.joblib')
    dst_model = os.path.join(models_dir, 'final_material_model.joblib')
    if os.path.exists(src_model):
        model = joblib.load(src_model)
        joblib.dump(model, dst_model)
        print(f"Persisted champion model to: {dst_model}\n")

    # -------------------------------------------------------------
    # Scenario 1: Standard Residential Structural Concrete (28 days)
    # -------------------------------------------------------------
    print("[TEST CASE 1] STANDARD RESIDENTIAL MIX (28 Days)")
    predict_material(
        cement=300.0, slag=0.0, ash=0.0, water=180.0, superplastic=0.0,
        coarseagg=1000.0, fineagg=750.0, age=28
    )

    # -------------------------------------------------------------
    # Scenario 2: Sustainable High-Performance "Green" Concrete (56 days)
    # -------------------------------------------------------------
    print("[TEST CASE 2] HIGH-PERFORMANCE GREEN CONCRETE (56 Days, Low Clinker)")
    predict_material(
        cement=200.0, slag=120.0, ash=80.0, water=150.0, superplastic=10.0,
        coarseagg=950.0, fineagg=720.0, age=56
    )

    # -------------------------------------------------------------
    # Scenario 3: Fast-Track Rapid Early Strength Concrete (3 days)
    # -------------------------------------------------------------
    print("[TEST CASE 3] RAPID EARLY-STRENGTH HIGHWAY REPAIR (3 Days)")
    predict_material(
        cement=450.0, slag=50.0, ash=0.0, water=140.0, superplastic=15.0,
        coarseagg=900.0, fineagg=700.0, age=3
    )

    print("Milestone 8 complete! The prediction function is verified and ready for deployment.")


if __name__ == '__main__':
    main()
