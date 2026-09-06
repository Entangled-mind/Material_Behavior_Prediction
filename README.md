# Material Behavior Prediction Using Machine Learning

An end-to-end, portfolio-quality machine learning project demonstrating how data-driven models can estimate the mechanical behavior of materials from their chemical formulation and environmental/curing conditions.

---

## 1. Problem Statement

In civil and structural materials engineering, knowing the compressive strength of a material formulation is critical for life safety, structural durability, and sustainable construction. 

Traditionally, evaluating a new concrete mix requires:
1. Batching and weighing raw materials in a laboratory.
2. Casting standardized cylinders into physical molds.
3. Submerging specimens in temperature-controlled water baths to cure for **7, 28, or up to 365 days**.
4. Destructively crushing each cylinder in a hydraulic compression frame to measure the failure load.

This manual experimental workflow is:
- **Extremely slow:** Waiting weeks to months for curing slows construction timelines.
- **Resource-intensive:** Physical trials consume energy, virgin raw materials, and laboratory labor.

### The Machine Learning Solution
By training machine learning models on validated experimental data, we map the non-linear relationship:

$$\text{Material Formulation (Composition)} + \text{Processing Conditions (Curing Time)} \xrightarrow{\text{Machine Learning}} \text{Predicted Mechanical Strength (MPa)}$$

---

## 2. Dataset Overview

For Version 1, we use the gold-standard **Concrete Compressive Strength Benchmark Dataset** (Prof. I-Cheng Yeh, 1998, *Cement and Concrete Research*), hosted on the UCI Machine Learning Repository:

- **Data Source:** Peer-reviewed physical laboratory test batches.
- **Total Samples:** 1,030 experimental formulations (reduced to 1,005 after removing exact replicate batches).
- **Attributes:** 9 total (8 input features + 1 continuous target).
- **Missing Values:** Exactly **0** missing values across all columns.
- **What Each Row Represents:** Exactly one physical cylinder specimen mixed with specific ingredient quantities ($kg/m^3$), cured under standard conditions for a specific number of days, and tested to mechanical failure.

### Feature Specification Table

| Column Name | Role | Units | Description | Min | Mean | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cement` | Input ($X_1$) | $\text{kg/m}^3$ | Portland cement (primary hydraulic binder) | 102.0 | 281.2 | 540.0 |
| `slag` | Input ($X_2$) | $\text{kg/m}^3$ | Blast furnace slag (recycled secondary binder) | 0.0 | 73.9 | 359.4 |
| `ash` | Input ($X_3$) | $\text{kg/m}^3$ | Fly ash (pozzolanic coal byproduct) | 0.0 | 54.2 | 200.1 |
| `water` | Input ($X_4$) | $\text{kg/m}^3$ | Mixing water (activates hydration reaction) | 121.8 | 181.6 | 247.0 |
| `superplastic` | Input ($X_5$) | $\text{kg/m}^3$ | Superplasticizer (chemical water-reducing admixture) | 0.0 | 6.2 | 32.2 |
| `coarseagg` | Input ($X_6$) | $\text{kg/m}^3$ | Coarse aggregate (crushed stone/gravel skeleton) | 801.0 | 972.9 | 1145.0 |
| `fineagg` | Input ($X_7$) | $\text{kg/m}^3$ | Fine aggregate (natural or manufactured sand) | 594.0 | 773.6 | 992.6 |
| `age` | Input ($X_8$) | Days | Curing duration prior to compression testing | 1 | 45.7 | 365 |
| **`strength`** | **Target ($y$)** | **MPa** | **Compressive strength under destructive mechanical load** | **2.33** | **35.82** | **82.60** |

---

## 3. Project Architecture & Modular Layout

```
Material_Behavior_Prediction/
│
├── data/
│   └── material_data.csv        # 1,030 experimental benchmark samples
│
├── notebooks/
│   └── material_prediction.ipynb # Interactive end-to-end Jupyter Notebook
│
├── src/
│   ├── data_understanding.py    # Milestone 1: Automated hygiene & schema audit
│   ├── eda.py                   # Milestone 2: 6 Domain visualizations (Abrams' law, kinetics)
│   ├── preprocessing.py         # Milestone 3: Duplicate removal, 80/20 split, zero-leakage scaling
│   ├── train_model.py           # Milestone 4: Baseline Linear Regression training & diagnostics
│   ├── evaluate_model.py        # Milestone 5: 4-Model benchmark comparison & ranking
│   ├── interpret_model.py       # Milestone 6: Tree MDI & test permutation feature importances
│   ├── error_analysis.py        # Milestone 7: Tolerance bands & worst prediction root-cause audit
│   └── predict_new_material.py  # Milestone 8: Inference interface & structural classification
│
├── models/                      # Serialized .joblib models, scalers & error audit tables
├── visualizations/              # 16 High-resolution diagnostic & exploratory charts
├── Milestone_1_Guide.pdf        # Illustrated technical documentation PDFs (Milestones 1-8)
├── ...
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── .gitignore                   # Git exclusion rules
```

---

## 4. End-to-End Methodological Approach

```
Raw Experimental Data
         │
         ▼
[Milestone 1: Data Understanding] ──► Schema verification, zero-missing checks, physical bounds
         │
         ▼
[Milestone 2: Exploratory EDA]   ──► Target bell curve, Abrams' Law verification, hydration kinetics
         │
         ▼
[Milestone 3: Preprocessing]     ──► Duplicate removal (1030 -> 1005), 80/20 split, anti-leakage scaling
         │
         ▼
[Milestone 4: Baseline Model]    ──► OLS Linear Regression (R² = 58%, MAE = 8.90 MPa)
         │
         ▼
[Milestone 5: Model Comparison]  ──► Benchmarked 4 algorithms: Trees, Random Forest, Gradient Boosting
         │
         ▼
[Milestone 6: Interpretation]    ──► Discovered 'age' + 'cement' + 'water' govern 75.8% of strength
         │
         ▼
[Milestone 7: Error Analysis]    ──► 75.6% within ±5 MPa, 95.5% within ±10 MPa; audited tail errors
         │
         ▼
[Milestone 8: New Inference]     ──► predict_material() deployed for custom mix design
```

---

## 5. Machine Learning Models Explored

| Model | Why It Was Evaluated | Mathematical Rationale |
| :--- | :--- | :--- |
| **1. Linear Regression** | Baseline Benchmark | Simple, transparent, and establishes the minimum performance floor to beat. |
| **2. Decision Tree** | Single Non-Linear Model | Captures orthogonal physical thresholds (e.g. $w/c < 0.45$), but produces jagged step boundaries. |
| **3. Random Forest** | Bagging Ensemble (100 trees) | Averages 100 de-correlated bootstrap trees to eliminate individual tree variance and smooth predictions. |
| **4. Gradient Boosting &starf;** | Boosting Ensemble (100 trees) | **Champion Model.** Sequentially corrects residual errors, optimizing non-linear multi-component interactions. |

---

## 6. Model Comparison & Benchmark Results

All models were evaluated on the strictly held-out **201 unseen test samples** ($20\%$ of data):

| Model | Train $R^2$ | Test $R^2$ | Test MAE | Test RMSE | Generalization Gap | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.6099 | 0.5802 | 8.90 MPa | 11.19 MPa | 0.0297 | Underfits non-linear physics |
| **Decision Tree** | 0.8641 | 0.7885 | 5.95 MPa | 7.94 MPa | 0.0755 | Moderate variance |
| **Random Forest** | 0.9824 | 0.9069 | 3.55 MPa | 5.27 MPa | 0.0755 | Strong generalization |
| **Gradient Boosting &starf;** | **0.9699** | **0.9198** | **3.45 MPa** | **4.89 MPa** | **0.0501** | **Champion Model** |

### Key Performance Findings:
- **61.2% Error Reduction:** Moving from the linear baseline to Gradient Boosting slashed average error from **8.90 MPa down to 3.45 MPa**.
- **92.0% Variance Explained:** Gradient Boosting captures nearly all mechanical behavior with an $R^2$ of **0.9198**.
- **Minimal Overfitting:** The generalization gap between training and testing $R^2$ is only **0.0501**, proving robust real-world generalization.

---

## 7. Model Interpretation & Materials Science Reasoning

Machine learning models should not be black boxes. We analyzed feature contributions using **Tree MDI Importance** and **Test Set Permutation Importance** (10 repeats):

| Feature | Relative Importance | Test Permutation Drop ($\Delta R^2$) | Physical & Chemical Role in Concrete |
| :--- | :--- | :--- | :--- |
| **`age`** | **37.3%** | **-0.7147** | **Hydration kinetics:** Curing time allows microscopic C-S-H crystal networks to densify. |
| **`cement`** | **28.9%** | **-0.5357** | **Primary binder:** Provides the reactive calcium and silicates forming the solid matrix. |
| **`water`** | **9.6%** | **-0.1800** | **Abrams' Law:** Governs capillary porosity. Excess water evaporates, leaving weak pores. |
| **`superplastic`**| 8.9% | -0.0722 | **Admixture:** Deflocculates cement particles, reducing water need while maintaining fluidity. |
| **`slag`** | 7.9% | -0.1436 | **Latent binder:** Recycled byproduct providing secondary hydration over time. |
| **`fineagg`** | 4.6% | -0.0441 | **Fine skeleton:** Sand particles filling microscopic voids between gravel stones. |
| **`coarseagg`** | 1.6% | -0.0128 | **Coarse skeleton:** Crushed gravel providing bulk stiffness against deformation. |
| **`ash`** | 1.2% | -0.0028 | **Pozzolanic binder:** Consumes calcium hydroxide to form durable secondary hydrates. |

> **Major Scientific Discovery:** Together, **`age` (37.3%) + `cement` (28.9%) + `water` (9.6%) account for 75.8% of total strength**. The machine learning algorithm independently verified the two cornerstone laws of concrete engineering: **Abrams' Law ($w/c$ ratio)** and **logarithmic curing kinetics ($t$)**!

---

## 8. Error Analysis & Limitations

### Practical Engineering Tolerances (ASTM C39 Standards)
- **Within ±2.5 MPa:** **49.8%** of test formulations (tight laboratory gauge precision).
- **Within ±5.0 MPa:** **75.6%** of test formulations (standard field batch-to-batch acceptance).
- **Within ±10.0 MPa:** **95.5%** of test formulations (broad structural safety envelope).

### Transparent Audit of Largest Errors
| Sample ID | Cement | Water | Superplastic | Age | Actual Strength | Predicted Strength | Residual Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #381 | 315.0 | 145.0 | 5.9 | 28 days | 81.75 MPa | 57.24 MPa | +24.51 MPa (underpredict) |
| #404 | 275.0 | 162.0 | 10.4 | 28 days | 76.24 MPa | 55.06 MPa | +21.18 MPa (underpredict) |
| #395 | 200.0 | 190.0 | 0.0 | 28 days | 49.25 MPa | 30.91 MPa | +18.34 MPa (underpredict) |
| #828 | 522.0 | 146.0 | 0.0 | 28 days | 74.99 MPa | 60.65 MPa | +14.34 MPa (underpredict) |

### Honest Project Limitations
1. **Ultra-High Strength Boundary (>70 MPa):**
   The model underpredicts mixes exceeding 70 MPa because high-strength concrete represents only ~5% of training data. Decision tree terminal leaves average out around 55–62 MPa and cannot extrapolate to extreme tail targets.
2. **Dataset Coverage & Curing Conditions:**
   All 1,030 samples were cured in standard moist/water rooms. Environmental stressors (freeze-thaw cycling, thermal shock, sulfate attack, alkali-silica reactions) are not present in this dataset.
3. **Material Diversity:**
   This model applies specifically to concrete mixtures. It cannot predict the behavior of steel, polymers, or ceramics without training data from those material classes.

---

## 9. How to Use the Prediction Interface

You can import `predict_material` from `src/predict_new_material.py` to evaluate new formulations:

```python
from src.predict_new_material import predict_material

# Predict compressive strength for an eco-friendly ternary blend
predicted_strength = predict_material(
    cement=220.0,       # kg/m³
    slag=110.0,         # kg/m³
    ash=75.0,           # kg/m³
    water=155.0,        # kg/m³
    superplastic=9.0,   # kg/m³
    coarseagg=960.0,    # kg/m³
    fineagg=740.0,      # kg/m³
    age=28              # days
)

print(f"Predicted Compressive Strength: {predicted_strength:.2f} MPa")
# Output: ~47.8 MPa (Heavy Commercial Structural Concrete)
```

---

## 10. Future Work & Roadmap

While Version 1 establishes a rock-solid Python ML foundation, future iterations can expand into:
- **Interactive Web Interface:** Deploying a real-time **Streamlit** dashboard for materials engineers to adjust sliders for cement, water, and age and see instant strength predictions.
- **Relational Storage:** Creating an **SQL database** (PostgreSQL / SQLite) to store thousands of commercial batch trial tickets and mix designs.
- **BI Visualizations:** Exporting predictions into **Power BI** or **Tableau** to visualize company-wide concrete quality control metrics.
- **Broader Environmental Variables:** Integrating curing temperature, relative humidity, and carbonation exposure datasets.

---

## 11. Important Scientific Rule & Disclaimer

> [!IMPORTANT]
> **Domain of Applicability:** This model estimates compressive strength specifically for concrete formulations whose ingredient proportions and curing ages fall within or reasonably near the experimental bounds of the Yeh (1998) dataset.
> 
> **Scientific Integrity:** A model prediction is a **statistical estimate** derived from learned empirical patterns. It must **never be presented as an experimental fact**. In structural engineering, model predictions must always be verified by standard destructive compression tests (such as ASTM C39 or BS EN 12390) before being specified in load-bearing, life-critical infrastructure.
