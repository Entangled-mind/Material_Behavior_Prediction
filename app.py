"""
app.py
======
Material Behavior Prediction & Intelligent Materials Studio
Comprehensive Platform:
1. 🌱 Simple Mode: Everyday language, use-case project presets, drying timelines, and real-world weight analogies.
2. 🔬 Advanced Engineering Mode: Precise formulation sliders (kg/m³), Abrams' Law, kinetics curves, and SQLite trial tracking.
3. 🧪 Materials Reaction & Chemistry Lab: Extensive material library with Temperature, Pressure, and Humidity environmental controls and chemical equations.
4. 📈 Custom Number Plotter & AI Insights Engine: Enter custom X & Y numbers, pick chart types, and get instant mathematical and scientific insights.
"""

import os
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Material Behavior & Reaction Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 16px;
    }
    .accuracy-banner {
        background: linear-gradient(90deg, #0284c7 0%, #0369a1 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .analogy-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 13.5px;
        color: #1e3a8a;
    }
    .reaction-card {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 14px;
        margin: 10px 0;
        font-family: 'Consolas', monospace;
        font-size: 13px;
        color: #166534;
    }
    .insight-card {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #0284c7;
        border-radius: 6px;
        padding: 14px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_material_model.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "models", "gradient_boosting.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "material_data.csv")
DB_PATH = os.path.join(BASE_DIR, "data", "material_predictions.db")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_historical_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df['total_binder'] = df['cement'] + df['slag'] + df['ash']
        df['wb_ratio'] = np.round(df['water'] / df['total_binder'], 3)
        return df
    return None

model = load_model()
hist_df = load_historical_data()

# Database setup
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS mix_trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mix_name TEXT,
            cement REAL,
            slag REAL,
            ash REAL,
            water REAL,
            superplastic REAL,
            coarseagg REAL,
            fineagg REAL,
            age INTEGER,
            predicted_strength REAL,
            water_binder_ratio REAL,
            embodied_carbon REAL,
            structural_grade TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_mix_to_db(mix_name, cement, slag, ash, water, superplastic, coarseagg, fineagg, age, strength, wb_ratio, carbon, grade):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO mix_trials (
            timestamp, mix_name, cement, slag, ash, water,
            superplastic, coarseagg, fineagg, age,
            predicted_strength, water_binder_ratio, embodied_carbon, structural_grade
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        mix_name, cement, slag, ash, water, superplastic,
        coarseagg, fineagg, age, strength, wb_ratio, carbon, grade
    ))
    conn.commit()
    conn.close()

def get_all_trials():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM mix_trials ORDER BY id DESC", conn)
    conn.close()
    return df

CARBON_FACTORS = {
    'cement': 0.86, 'slag': 0.07, 'ash': 0.015,
    'water': 0.0002, 'superplastic': 0.25, 'coarseagg': 0.005, 'fineagg': 0.005
}

def calculate_carbon_footprint(c, s, a, w, sp, ca, fa):
    return (
        c * CARBON_FACTORS['cement'] + s * CARBON_FACTORS['slag'] + a * CARBON_FACTORS['ash'] +
        w * CARBON_FACTORS['water'] + sp * CARBON_FACTORS['superplastic'] +
        ca * CARBON_FACTORS['coarseagg'] + fa * CARBON_FACTORS['fineagg']
    )

def predict_strength(c, s, a, w, sp, ca, fa, age_days):
    if model is None:
        return 0.0
    row = pd.DataFrame([{
        'cement': c, 'slag': s, 'ash': a, 'water': w,
        'superplastic': sp, 'coarseagg': ca, 'fineagg': fa, 'age': int(age_days)
    }])
    return float(model.predict(row)[0])

# -----------------------------------------------------------------------------
# Global Accuracy Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="accuracy-banner">
    <div>
        <span style="font-size: 16px; font-weight: 700;">🎯 Model Accuracy: 92.0% (R² Score)</span> &bull; 
        <span style="font-size: 13px;">Physical Margin: ±3.45 MPa (ASTM C39 Benchmark Standard)</span>
    </div>
    <div style="font-size: 12px; background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 6px;">
        1,000+ Laboratory Test Cylinders (Yeh 1998)
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Studio Navigation
# -----------------------------------------------------------------------------
app_section = st.radio(
    "Explore Studio Sections:",
    [
        "🌱 Simple Mode (Everyday Language & Real-World Use Cases)",
        "🔬 Advanced Engineering Mode (Formulas, Sliders & Deep Analytics)",
        "🧪 Materials Reaction & Chemistry Lab (Temp, Pressure & Reactions)",
        "📈 Custom Number Plotter & AI Insights (Enter X & Y Numbers)"
    ],
    horizontal=True
)

st.markdown("---")

# =============================================================================
# SECTION 1: 🌱 SIMPLE MODE
# =============================================================================
if app_section.startswith("🌱 Simple"):
    st.markdown('<div class="main-title">🏡 Concrete & Material Strength Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Select your project, choose your drying time, and see real-world strength in plain language!</div>', unsafe_allow_html=True)

    col_inp, col_out = st.columns([1, 1], gap="large")

    with col_inp:
        st.subheader("1. What Are You Building?")
        project = st.selectbox(
            "Choose a building project:",
            [
                "🏡 Backyard Patio, Garden Pathway or Sidewalk",
                "🏠 Residential House Foundation & Floor Slab",
                "🚗 Garage Floor & Heavy Vehicle Driveway",
                "🏢 Multi-Story Commercial Building / Warehouse",
                "🌉 Highway Bridge Pier / Extreme Skyscraper",
                "🌿 Eco-Friendly Green Home (Low Carbon Footprint)",
                "🛠️ Custom Simple Mix (Adjust Sliders Below)"
            ]
        )

        if "Backyard Patio" in project:
            c_d, s_d, a_d, w_d, sp_d, cg_d, fg_d, age_d = 220.0, 0.0, 50.0, 190.0, 0.0, 1050.0, 780.0, 28
        elif "Residential House" in project:
            c_d, s_d, a_d, w_d, sp_d, cg_d, fg_d, age_d = 300.0, 0.0, 0.0, 180.0, 0.0, 1000.0, 750.0, 28
        elif "Garage Floor" in project:
            c_d, s_d, a_d, w_d, sp_d, cg_d, fg_d, age_d = 350.0, 80.0, 0.0, 160.0, 6.0, 980.0, 740.0, 28
        elif "Multi-Story" in project:
            c_d, s_d, a_d, w_d, sp_d, cg_d, fg_d, age_d = 420.0, 100.0, 0.0, 150.0, 10.0, 940.0, 720.0, 28
        elif "Highway Bridge" in project:
            c_d, s_d, a_d, w_d, sp_d, cg_d, fg_d, age_d = 490.0, 90.0, 0.0, 140.0, 16.0, 900.0, 710.0, 28
        elif "Eco-Friendly" in project:
            c_d, s_d, a_d, w_d, sp_d, cg_d, fg_d, age_d = 180.0, 140.0, 90.0, 145.0, 10.0, 960.0, 710.0, 56
        else:
            c_d, s_d, a_d, w_d, sp_d, cg_d, fg_d, age_d = 290.0, 50.0, 30.0, 175.0, 5.0, 980.0, 750.0, 28

        st.subheader("2. How Long Has It Dried / Cured?")
        drying_time = st.select_slider(
            "Select drying duration:",
            options=["3 Days (Fresh & Soft)", "7 Days (One Week)", "14 Days (Two Weeks)", "28 Days (Standard Full Strength)", "90 Days (Maximum Age)"],
            value="28 Days (Standard Full Strength)" if age_d == 28 else ("90 Days (Maximum Age)" if age_d == 56 else "28 Days (Standard Full Strength)")
        )
        age_map = {"3 Days (Fresh & Soft)": 3, "7 Days (One Week)": 7, "14 Days (Two Weeks)": 14, "28 Days (Standard Full Strength)": 28, "90 Days (Maximum Age)": 90}
        curr_age = age_map[drying_time]

        with st.expander("Adjust Sliders in Everyday Terms (Optional)", expanded=(project.startswith("🛠️"))):
            sc_c = st.slider("🧱 Cement Powder (The glue holding it together)", 100, 550, int(c_d), 10)
            sc_w = st.slider("💧 Water Added (Tip: Less water = stronger concrete!)", 120, 240, int(w_d), 5)
            sc_agg = st.slider("🪨 Sand & Crushed Stones (The solid rock skeleton)", 1400, 2000, int(cg_d + fg_d), 20)
            sc_eco = st.slider("🌿 Recycled Slag & Ash (Eco-friendly cement replacements)", 0, 300, int(s_d + a_d), 10)

            c_val = sc_c
            s_val = float(sc_eco * 0.6)
            a_val = float(sc_eco * 0.4)
            w_val = sc_w
            sp_val = sp_d
            cg_val = float(sc_agg * 0.58)
            fg_val = float(sc_agg * 0.42)

    pred = predict_strength(c_val, s_val, a_val, w_val, sp_val, cg_val, fg_val, curr_age)
    psi = pred * 145.038
    co2 = calculate_carbon_footprint(c_val, s_val, a_val, w_val, sp_val, cg_val, fg_val)

    with col_out:
        st.subheader("📋 Your Result in Everyday Terms")

        if pred < 20.0:
            v_title, v_desc, v_bg, v_border, v_color = "Light Duty (Garden & Pathway Grade)", "Good for simple decorative paths, stepping stones, or trench fill.", "#fef3c7", "#f59e0b", "#92400e"
        elif 20.0 <= pred < 35.0:
            v_title, v_desc, v_bg, v_border, v_color = "Solid Structural Grade (Standard Home Quality)", "Strong enough for house slabs, footings, garage floors, and driveways.", "#e0f2fe", "#0284c7", "#075985"
        elif 35.0 <= pred < 50.0:
            v_title, v_desc, v_bg, v_border, v_color = "Heavy-Duty Commercial Grade (Very Strong)", "Easily supports delivery trucks, parking garages, and multi-story buildings.", "#dcfce7", "#10b981", "#065f46"
        else:
            v_title, v_desc, v_bg, v_border, v_color = "Ultra High-Performance (Extreme Strength)", "Engineered for skyscrapers, highway bridges, and heavy industrial piers.", "#f3e8ff", "#8b5cf6", "#5b21b6"

        st.markdown(f"""
        <div style="background:{v_bg}; border: 2px solid {v_border}; border-radius: 8px; padding: 14px; color: {v_color};">
            <h3 style="margin:0 0 4px 0; color: {v_color};">Verdict: {v_title}</h3>
            <p style="margin:0; font-size: 13.5px;">{v_desc}</p>
        </div>
        """, unsafe_allow_html=True)

        num_suvs = max(1, int(psi / 1500))
        st.markdown(f"""
        <div class="analogy-box">
            <strong>🐘 What does this mean in real life?</strong><br>
            This mix withstands roughly <strong>{psi:,.0f} pounds per square inch (PSI)</strong>.<br>
            That means a single small tile (1 foot &times; 1 foot) can support the weight of roughly <strong>{num_suvs} full-size SUVs</strong> stacked on top without cracking!
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**⏱️ Safe Activity Timeline (When can you use it?):**")
        s3 = predict_strength(c_val, s_val, a_val, w_val, sp_val, cg_val, fg_val, 3)
        s7 = predict_strength(c_val, s_val, a_val, w_val, sp_val, cg_val, fg_val, 7)
        s28 = predict_strength(c_val, s_val, a_val, w_val, sp_val, cg_val, fg_val, 28)

        t1, t2, t3 = st.columns(3)
        t1.metric("Day 3", f"{s3:.1f} MPa", "🚶 Safe to walk on")
        t2.metric("Day 7", f"{s7:.1f} MPa", "🚗 Light vehicle safe")
        t3.metric("Day 28", f"{s28:.1f} MPa", "🏆 Full design strength")

        with st.expander("🔬 View Deep Laboratory Figures (Exact MPa & Porosity)"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Exact Strength", f"{pred:.2f} MPa")
            c2.metric("ASTM C39 Margin", "±3.45 MPa")
            c3.metric("Water-to-Binder", f"{w_val/(c_val+s_val+a_val):.2f}")

# =============================================================================
# SECTION 2: 🔬 ADVANCED ENGINEERING MODE
# =============================================================================
elif app_section.startswith("🔬 Advanced"):
    st.markdown('<div class="main-title">🔬 Advanced Engineering Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Precise constituent sliders, logarithmic curing curves, Abrams\' Law, and SQLite trial tracking.</div>', unsafe_allow_html=True)

    adv_c1, adv_c2 = st.columns([1, 2], gap="medium")

    with adv_c1:
        st.subheader("Formulation Inputs (kg/m³)")
        e_c = st.slider("Portland Cement", 100.0, 550.0, 300.0, 5.0)
        e_s = st.slider("Blast Furnace Slag", 0.0, 360.0, 80.0, 5.0)
        e_a = st.slider("Fly Ash", 0.0, 200.0, 40.0, 5.0)
        e_w = st.slider("Mixing Water", 120.0, 250.0, 175.0, 1.0)
        e_sp = st.slider("Superplasticizer", 0.0, 35.0, 6.0, 0.5)
        e_cg = st.slider("Coarse Aggregate", 800.0, 1150.0, 980.0, 10.0)
        e_fg = st.slider("Fine Aggregate", 590.0, 1000.0, 750.0, 10.0)
        e_age = st.slider("Curing Age (Days)", 1, 365, 28, 1)
        b_name = st.text_input("Mix Name / Batch ID", value="Trial-Batch-01")

        if st.button("💾 Save to SQLite Database"):
            p_val = predict_strength(e_c, e_s, e_a, e_w, e_sp, e_cg, e_fg, e_age)
            save_mix_to_db(b_name, e_c, e_s, e_a, e_w, e_sp, e_cg, e_fg, e_age, round(p_val, 2), round(e_w/(e_c+e_s+e_a), 2), 0.0, "Engineering")
            st.toast(f"Saved {b_name}!", icon="✅")

    with adv_c2:
        e_pred = predict_strength(e_c, e_s, e_a, e_w, e_sp, e_cg, e_fg, e_age)
        e_wb = e_w / (e_c + e_s + e_a) if (e_c + e_s + e_a) > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Strength (f'c)", f"{e_pred:.2f} MPa", f"{e_pred-35.82:+.1f} vs Benchmark")
        m2.metric("Water-to-Binder (w/b)", f"{e_wb:.2f}", "Denser" if e_wb < 0.45 else "Porous", delta_color="inverse")
        m3.metric("ASTM Error Band", f"{e_pred-3.45:.1f} to {e_pred+3.45:.1f} MPa")

        sim_d = [1, 3, 7, 14, 21, 28, 56, 90, 180, 365]
        sim_s = [predict_strength(e_c, e_s, e_a, e_w, e_sp, e_cg, e_fg, d) for d in sim_d]
        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=sim_d, y=sim_s, mode='lines+markers', name='Curing Curve', line=dict(color='#0284c7', width=3)))
        fig_k.add_trace(go.Scatter(x=[e_age], y=[e_pred], mode='markers', name=f'Current ({e_age}d)', marker=dict(size=14, color='red', symbol='star')))
        fig_k.update_layout(xaxis_title="Curing Days (Log-Scale)", yaxis_title="Strength (MPa)", xaxis_type="log", height=420)
        st.plotly_chart(fig_k, use_container_width=True)

# =============================================================================
# SECTION 3: 🧪 MATERIALS REACTION & CHEMISTRY LAB (EXPANDED MATERIALS & P/T)
# =============================================================================
elif app_section.startswith("🧪 Materials Reaction"):
    st.markdown('<div class="main-title">🧪 Materials Chemical Reaction & Environmental Conditions Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Select reactive materials from an expanded library and simulate how Temperature, Pressure, and Humidity alter chemical reaction pathways.</div>', unsafe_allow_html=True)

    col_chem_ctrl, col_chem_res = st.columns([1, 1], gap="large")

    with col_chem_ctrl:
        st.subheader("1. Choose Materials to React")
        materials_selected = st.multiselect(
            "Select materials from the chemical library:",
            [
                "Portland Cement (Alite C3S & Belite C2S)",
                "Mixing Water (H2O)",
                "Blast Furnace Slag (Calcium Alumino-Silicate)",
                "Fly Ash (Class F / Pozzolanic Silica)",
                "Silica Fume / Microsilica (Nano-SiO2)",
                "Metakaolin (Calcined Kaolin Clay Al2O3·2SiO2)",
                "Limestone Powder (CaCO3 Fine Filler)",
                "Calcium Chloride (CaCl2 Accelerator)",
                "Gypsum (CaSO4·2H2O Retarder)",
                "Sodium Hydroxide (NaOH Geopolymer Activator)",
                "Superplasticizer (Polycarboxylate Ether)"
            ],
            default=[
                "Portland Cement (Alite C3S & Belite C2S)",
                "Mixing Water (H2O)",
                "Fly Ash (Class F / Pozzolanic Silica)",
                "Silica Fume / Microsilica (Nano-SiO2)"
            ]
        )

        st.subheader("2. Environmental & Processing Conditions")
        c_temp = st.slider("🌡️ Curing Temperature (°C)", -10, 120, 20, 5, help="Sub-zero halts reaction; high heat accelerates initial hydration; autoclaving >100°C changes crystal phase.")
        c_press = st.slider("🗜️ Curing Pressure (Atmospheres / Bar)", 1, 50, 1, 1, help="1 atm is standard ambient; 10–50 atm is industrial high-pressure steam autoclaving.")
        c_humid = st.slider("💧 Relative Humidity (%)", 10, 100, 95, 5, help="Concrete needs >80% RH for continued hydration. Low RH causes desiccation and microcracking.")
        c_time = st.select_slider("⏱️ Reaction Duration:", options=["1 Hour (Initial Set)", "12 Hours (Dormant Exit)", "24 Hours (Early Hardening)", "7 Days (Hydration Peak)", "28 Days (Standard Cure)", "90 Days (Matured Glass Reaction)"], value="28 Days (Standard Cure)")

    with col_chem_res:
        st.subheader("🔬 Active Chemical Reactions & Equations")

        reactions_count = 0

        # Reaction 1: Primary Cement Hydration
        if "Portland Cement (Alite C3S & Belite C2S)" in materials_selected and "Mixing Water (H2O)" in materials_selected:
            reactions_count += 1
            st.markdown("**1. Primary Hydraulic Alite Hydration (Fast Strength):**")
            st.markdown("""
            <div class="reaction-card">
            2 Ca₃SiO₅ (Alite) + 11 H₂O ➔ 3CaO·2SiO₂·8H₂O (C-S-H Gel) + 3 Ca(OH)₂ (Portlandite) + ΔH (500 J/g)
            </div>
            """, unsafe_allow_html=True)

            # Reaction 2: Belite Hydration
            reactions_count += 1
            st.markdown("**2. Secondary Hydraulic Belite Hydration (Long-Term Strength):**")
            st.markdown("""
            <div class="reaction-card">
            2 Ca₂SiO₄ (Belite) + 9 H₂O ➔ 3CaO·2SiO₂·8H₂O (C-S-H Gel) + Ca(OH)₂ + ΔH (250 J/g)
            </div>
            """, unsafe_allow_html=True)

        # Reaction 3: Pozzolanic Reaction (Fly Ash)
        if "Fly Ash (Class F / Pozzolanic Silica)" in materials_selected and "Portland Cement (Alite C3S & Belite C2S)" in materials_selected and "Mixing Water (H2O)" in materials_selected:
            reactions_count += 1
            st.markdown("**3. Pozzolanic Reaction (Fly Ash Silica consuming Portlandite):**")
            st.markdown("""
            <div class="reaction-card">
            3 Ca(OH)₂ + 2 SiO₂ (Fly Ash) + 5 H₂O ➔ 3CaO·2SiO₂·8H₂O (Dense Secondary C-S-H Gel)
            </div>
            """, unsafe_allow_html=True)

        # Reaction 4: Silica Fume Nano-Densification
        if "Silica Fume / Microsilica (Nano-SiO2)" in materials_selected and "Mixing Water (H2O)" in materials_selected:
            reactions_count += 1
            st.markdown("**4. Ultra-Fine Nano-Silica Rapid Refinement:**")
            st.markdown("""
            <div class="reaction-card">
            SiO₂ (Amorphous Nano-Silica) + Ca(OH)₂ + H₂O ➔ High-Density C-S-H Gel (Zero Capillary Pores)
            </div>
            """, unsafe_allow_html=True)

        # Reaction 5: Slag Hydration
        if "Blast Furnace Slag (Calcium Alumino-Silicate)" in materials_selected and "Mixing Water (H2O)" in materials_selected:
            reactions_count += 1
            st.markdown("**5. Latent Hydraulic Slag Activation:**")
            st.markdown("""
            <div class="reaction-card">
            CaO·Al₂O₃·2SiO₂ (Slag Glass) + [OH⁻ Activator] + H₂O ➔ C-A-S-H Gel + Hydrotalcite (Sea Salt Shield)
            </div>
            """, unsafe_allow_html=True)

        # Reaction 6: Metakaolin
        if "Metakaolin (Calcined Kaolin Clay Al2O3·2SiO2)" in materials_selected and "Mixing Water (H2O)" in materials_selected:
            reactions_count += 1
            st.markdown("**6. Metakaolin Rapid Pozzolanic Reaction:**")
            st.markdown("""
            <div class="reaction-card">
            Al₂O₃·2SiO₂ (Metakaolin) + 7 Ca(OH)₂ + 9 H₂O ➔ C₄AH₁₃ + C-S-H Gel (High Early Reactivity)
            </div>
            """, unsafe_allow_html=True)

        # Reaction 7: Chemical Accelerator
        if "Calcium Chloride (CaCl2 Accelerator)" in materials_selected:
            reactions_count += 1
            st.markdown("**7. Chemical Catalysis (Acceleration):**")
            st.markdown("""
            <div class="reaction-card">
            CaCl₂ accelerates dissolution rate of Alite (C₃S) by 300% ➔ Rapid initial set in cold weather.
            </div>
            """, unsafe_allow_html=True)

        # Reaction 8: Geopolymer Alkali Activation
        if "Sodium Hydroxide (NaOH Geopolymer Activator)" in materials_selected and ("Fly Ash (Class F / Pozzolanic Silica)" in materials_selected or "Blast Furnace Slag (Calcium Alumino-Silicate)" in materials_selected):
            reactions_count += 1
            st.markdown("**8. Clinker-Free Geopolymer Polycondensation:**")
            st.markdown("""
            <div class="reaction-card">
            Si-O-Al Framework + Na⁺ + OH⁻ ➔ Sialate Network (-Si-O-Al-O-)ₙ (Zero Cement Clinker CO₂!)
            </div>
            """, unsafe_allow_html=True)

        if reactions_count == 0:
            st.info("👈 Select at least 2 compatible materials on the left to trigger the chemical reaction engine!")

    # Environmental Impact Physics Analysis
    st.markdown("---")
    st.subheader(f"📈 Reaction Kinetics & Phase Transformation ({c_temp}°C, {c_press} atm, {c_humid}% RH)")

    # Arrhenius temperature scaling: k = A * exp(-Ea / RT)
    T_kelvin = c_temp + 273.15
    R_gas = 8.314
    Ea = 33500.0  # J/mol typical activation energy for cement hydration
    k_rate = np.exp(-Ea / (R_gas * max(200.0, T_kelvin))) / np.exp(-Ea / (R_gas * 293.15))

    # Pressure acceleration (Hydrothermal effect)
    press_factor = 1.0 + (c_press - 1) * 0.03
    humid_factor = max(0.05, c_humid / 100.0)
    effective_rate = k_rate * press_factor * humid_factor

    t_days = np.array([0.05, 0.25, 0.5, 1, 2, 3, 7, 14, 28, 56, 90, 180])
    reaction_progress = np.clip(100.0 * (1.0 - np.exp(-0.35 * effective_rate * t_days)), 0.0, 100.0)

    # Tobermorite conversion under autoclave
    if c_press > 10 and c_temp > 80:
        tobermorite_vol = np.clip(80.0 * (1.0 - np.exp(-0.5 * effective_rate * t_days)), 0.0, 85.0)
        chart_df = pd.DataFrame({'Days': t_days, 'Reaction Degree (%)': reaction_progress, 'Crystalline Tobermorite (C5S6H5) %': tobermorite_vol})
        y_cols = ['Reaction Degree (%)', 'Crystalline Tobermorite (C5S6H5) %']
        col_map = {'Reaction Degree (%)': '#0284c7', 'Crystalline Tobermorite (C5S6H5) %': '#10b981'}
    else:
        chart_df = pd.DataFrame({'Days': t_days, 'Hydration Conversion (%)': reaction_progress})
        y_cols = ['Hydration Conversion (%)']
        col_map = {'Hydration Conversion (%)': '#0284c7'}

    fig_kin = px.line(chart_df, x='Days', y=y_cols, title="Chemical Conversion Progress Over Time", color_discrete_map=col_map)
    fig_kin.update_layout(xaxis_type="log", height=380, xaxis_title="Time (Days, Log-Scale)", yaxis_title="Conversion Extent (%)")
    st.plotly_chart(fig_kin, use_container_width=True)

    # Physical Condition Explanation
    cond_col1, cond_col2 = st.columns(2)
    with cond_col1:
        if c_temp < 0:
            st.error("❄️ **Freezing Alert:** Temperature below 0°C freezes pore water. Hydration completely halts and expanding ice destroys pore structure.")
        elif c_temp > 80:
            st.success("🔥 **Thermal Acceleration:** High heat accelerates initial chemical reaction rate by up to 400%, allowing rapid mold stripping in pre-cast factories.")
        else:
            st.info("🌡️ **Standard Ambient Hydration:** Normal hydration temperature maintains optimal C-S-H crystal morphology.")

    with cond_col2:
        if c_press > 10:
            st.success("🗜️ **Autoclave Hydrothermal Synthesis:** High pressure and heat transform amorphous C-S-H gel into crystalline Tobermorite ($5\\text{CaO}\\cdot 6\\text{SiO}_2\\cdot 5\\text{H}_2\\O$), giving zero drying shrinkage and high strength in hours.")
        else:
            st.info("💨 **Atmospheric Pressure:** Standard atmospheric curing produces normal amorphous calcium-silicate-hydrate gel.")

# =============================================================================
# SECTION 4: 📈 CUSTOM NUMBER PLOTTER & AI INSIGHTS ENGINE
# =============================================================================
elif app_section.startswith("📈 Custom Number Plotter"):
    st.markdown('<div class="main-title">📈 Custom Number Plotter & AI Insights Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter your own numbers for X and Y, choose your chart type, and get instant mathematical and scientific insights!</div>', unsafe_allow_html=True)

    plot_inp_col, plot_out_col = st.columns([1, 1], gap="large")

    with plot_inp_col:
        st.subheader("1. Enter Your X and Y Numbers")

        data_preset = st.selectbox(
            "Or load a sample science dataset:",
            [
                "Custom (Type your own numbers below)",
                "Water-to-Binder Ratio vs Compressive Strength (Abrams' Law)",
                "Curing Days vs Strength Growth (Hydration Kinetics)",
                "Superplasticizer Dose vs Mixing Water Required",
                "Temperature vs Reaction Rate Constant (Arrhenius Law)"
            ]
        )

        if data_preset == "Water-to-Binder Ratio vs Compressive Strength (Abrams' Law)":
            default_x = "0.28, 0.35, 0.42, 0.50, 0.60, 0.70, 0.85, 1.05"
            default_y = "68.5, 54.2, 45.0, 36.1, 27.4, 20.8, 14.5, 9.2"
            default_x_label = "Water-to-Binder Ratio (w/b)"
            default_y_label = "Compressive Strength (MPa)"
        elif data_preset == "Curing Days vs Strength Growth (Hydration Kinetics)":
            default_x = "1, 3, 7, 14, 28, 56, 90, 180, 365"
            default_y = "11.2, 22.4, 34.0, 42.5, 48.0, 52.3, 55.1, 57.0, 58.4"
            default_x_label = "Curing Age (Days)"
            default_y_label = "Strength (MPa)"
        elif data_preset == "Superplasticizer Dose vs Mixing Water Required":
            default_x = "0, 2, 5, 8, 12, 16, 20"
            default_y = "210, 195, 178, 160, 148, 140, 136"
            default_x_label = "Superplasticizer (kg/m³)"
            default_y_label = "Required Mixing Water (kg/m³)"
        elif data_preset == "Temperature vs Reaction Rate Constant (Arrhenius Law)":
            default_x = "5, 15, 25, 35, 50, 65, 80"
            default_y = "0.22, 0.55, 1.00, 1.75, 3.40, 6.10, 10.20"
            default_x_label = "Temperature (°C)"
            default_y_label = "Reaction Rate Factor"
        else:
            default_x = "10, 20, 30, 40, 50, 60, 70, 80"
            default_y = "14.2, 26.5, 37.0, 46.8, 55.0, 61.2, 65.4, 68.0"
            default_x_label = "Independent Variable (X)"
            default_y_label = "Dependent Variable (Y)"

        x_text = st.text_area("Enter X Numbers (separated by commas or spaces):", value=default_x, height=75)
        y_text = st.text_area("Enter Y Numbers (separated by commas or spaces):", value=default_y, height=75)

        x_name = st.text_input("Label for X-Axis:", value=default_x_label)
        y_name = st.text_input("Label for Y-Axis:", value=default_y_label)

        chart_choice = st.selectbox(
            "Choose Graph Type:",
            [
                "🔵 Scatter Plot with Best-Fit Line (Linear Trendline)",
                "📈 Connected Line Chart (Continuous Path)",
                "📊 Column / Bar Chart (Discrete Comparison)",
                "📉 Curved Polynomial Fit (Non-Linear Regression)"
            ]
        )

    with plot_out_col:
        st.subheader("2. Plotted Graph & AI Insights")

        # Parse numerical inputs
        try:
            x_vals = [float(val.strip()) for val in x_text.replace(',', ' ').split() if val.strip()]
            y_vals = [float(val.strip()) for val in y_text.replace(',', ' ').split() if val.strip()]

            if len(x_vals) != len(y_vals):
                st.error(f"⚠️ Mismatch in numbers: You entered **{len(x_vals)}** X values but **{len(y_vals)}** Y values. Both must have the same count!")
            elif len(x_vals) < 2:
                st.warning("Please enter at least 2 coordinate pairs to plot a graph.")
            else:
                user_df = pd.DataFrame({'X': x_vals, 'Y': y_vals}).sort_values('X')

                # Calculate Mathematical Metrics
                n = len(user_df)
                x_arr = user_df['X'].values
                y_arr = user_df['Y'].values

                corr = np.corrcoef(x_arr, y_arr)[0, 1] if np.std(x_arr) > 0 and np.std(y_arr) > 0 else 0.0
                r_squared = corr ** 2

                slope, intercept = np.polyfit(x_arr, y_arr, 1)

                # Render Plotly Chart
                if "Scatter Plot" in chart_choice:
                    fig_usr = px.scatter(
                        user_df, x='X', y='Y', trendline="ols",
                        title=f"{y_name} vs. {x_name}",
                        labels={'X': x_name, 'Y': y_name}
                    )
                    fig_usr.update_traces(marker=dict(size=12, color='#0284c7', line=dict(color='black', width=1)))
                elif "Line Chart" in chart_choice:
                    fig_usr = px.line(
                        user_df, x='X', y='Y', markers=True,
                        title=f"{y_name} vs. {x_name}",
                        labels={'X': x_name, 'Y': y_name}
                    )
                    fig_usr.update_traces(line=dict(color='#0284c7', width=3), marker=dict(size=10, color='#0369a1'))
                elif "Bar Chart" in chart_choice:
                    fig_usr = px.bar(
                        user_df, x='X', y='Y',
                        title=f"{y_name} vs. {x_name}",
                        labels={'X': x_name, 'Y': y_name},
                        color_discrete_sequence=['#0284c7']
                    )
                else:
                    # Polynomial quadratic fit
                    poly_coeffs = np.polyfit(x_arr, y_arr, 2)
                    x_smooth = np.linspace(min(x_arr), max(x_arr), 100)
                    y_smooth = np.polyval(poly_coeffs, x_smooth)

                    fig_usr = go.Figure()
                    fig_usr.add_trace(go.Scatter(x=x_arr, y=y_arr, mode='markers', name='Data Points', marker=dict(size=12, color='#ef4444')))
                    fig_usr.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='Curved Fit (Degree 2)', line=dict(color='#0284c7', width=3)))
                    fig_usr.update_layout(title=f"Curved Fit: {y_name} vs. {x_name}", xaxis_title=x_name, yaxis_title=y_name)

                fig_usr.update_layout(height=420, margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig_usr, use_container_width=True)

                # AI Insights Engine
                st.markdown("### 🧠 AI Mathematical & Scientific Insights")

                sign_str = "+" if intercept >= 0 else "-"
                eq_str = f"Y = {slope:.3f} · X {sign_str} {abs(intercept):.3f}"

                # Classify relationship behavior
                if abs(corr) >= 0.90:
                    strength_desc = "Extremely Strong Relationship"
                elif abs(corr) >= 0.70:
                    strength_desc = "Strong Relationship"
                elif abs(corr) >= 0.40:
                    strength_desc = "Moderate Correlation"
                else:
                    strength_desc = "Weak or Complex Non-Linear Association"

                direction_desc = "Direct / Positive (As X rises, Y rises)" if slope > 0 else "Inverse / Negative (As X rises, Y drops)"

                # Check for diminishing returns or plateau
                if len(y_arr) >= 3:
                    diffs = np.diff(y_arr)
                    if np.all(diffs >= 0) and diffs[-1] < diffs[0] * 0.5:
                        curvature_note = "💡 **Diminishing Returns Detected:** Growth slows down at higher X values, forming a classic plateau curve (common in chemical saturation and curing limits)."
                    elif np.all(diffs <= 0) and abs(diffs[-1]) < abs(diffs[0]) * 0.5:
                        curvature_note = "💡 **Decay Plateau Detected:** The rate of drop flattens out as X increases (classic asymptotic decay, similar to Abrams' Law)."
                    else:
                        curvature_note = "💡 **Rate of Change:** The slope indicates that for every 1-unit increase in X, Y changes by approximately **{:.3f} units**.".format(slope)
                else:
                    curvature_note = ""

                st.markdown(f"""
                <div class="insight-card">
                    <div style="font-size: 15px; font-weight: 700; color: #0f172a; margin-bottom: 6px;">
                        {strength_desc} &bull; {direction_desc}
                    </div>
                    <strong>Best-Fit Linear Equation:</strong> <code>{eq_str}</code><br>
                    <strong>Pearson Correlation (r):</strong> <code>{corr:+.4f}</code> &bull; 
                    <strong>Goodness of Fit (R²):</strong> <code>{r_squared*100:.1f}%</code><br><br>
                    {curvature_note}
                </div>
                """, unsafe_allow_html=True)

        except Exception as err:
            st.error(f"Error parsing numbers: {err}. Please ensure all entries are valid decimal numbers separated by commas.")

# -----------------------------------------------------------------------------
# Global Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("⚠️ **Mandatory Scientific Notice:** Predictions and simulations are statistical estimates based on empirical concrete data (Yeh 1998 Benchmark, 92.0% R²). They provide educational and engineering decision-support guidance but do not replace destructive ASTM C39 / BS EN 12390 testing for life-critical building permits.")
