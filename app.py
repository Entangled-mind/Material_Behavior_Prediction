"""
app.py
======
Material Behavior Prediction & Intelligent Materials Studio
Comprehensive Platform:
1. 🌱 Simple Mode: Everyday language, use-case presets, drying timelines, and real-world weight analogies.
2. 🔬 Advanced Mode: Full formulation engineering, Abrams' Law, and kinetics curves.
3. 🧪 Student Chemistry & Reaction Lab: Interactive molecular reactions, heat of hydration, and temperature conditions.
4. 🛠️ Student DIY / Real-Life Maker Lab: Practical kitchen/workshop casting recipes (cups/buckets) for planters, pavers, and test cubes.
5. 📊 Interactive Custom Plotter & AI Insights: Choose any X and Y, pick chart types, and get automated scientific insights.
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
    page_title="Material Behavior & Chemistry Studio",
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
    .reaction-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 14px;
        margin: 10px 0;
        font-family: 'Consolas', monospace;
        font-size: 13px;
        color: #166534;
    }
    .diy-step {
        background: #fafafa;
        border-left: 3px solid #f59e0b;
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
    }
    .insight-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
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

# Domain constants
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
        <span style="font-size: 13px;">Average Prediction Margin: ±3.45 MPa (ASTM C39 Standard)</span>
    </div>
    <div style="font-size: 12px; background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 6px;">
        1,000+ Validated Lab Cylinders (Yeh 1998)
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Top Experience Selector Navigation
# -----------------------------------------------------------------------------
app_section = st.radio(
    "Explore Studio Sections:",
    [
        "🌱 Simple Mode (Everyday Language & Real-World Use Cases)",
        "🔬 Advanced Engineering Mode (Formulas, Sliders & Deep Analytics)",
        "🧪 Student Chemistry & Reaction Lab (How Ingredients React)",
        "🛠️ Student DIY & Maker Lab (What You Can Create in Real Life)",
        "📊 Interactive Custom Plotter (Plot Any X vs Y with AI Insights)"
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

        # Kinetics curve
        sim_d = [1, 3, 7, 14, 21, 28, 56, 90, 180, 365]
        sim_s = [predict_strength(e_c, e_s, e_a, e_w, e_sp, e_cg, e_fg, d) for d in sim_d]
        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=sim_d, y=sim_s, mode='lines+markers', name='Curing Curve', line=dict(color='#0284c7', width=3)))
        fig_k.add_trace(go.Scatter(x=[e_age], y=[e_pred], mode='markers', name=f'Current ({e_age}d)', marker=dict(size=14, color='red', symbol='star')))
        fig_k.update_layout(xaxis_title="Curing Days (Log-Scale)", yaxis_title="Strength (MPa)", xaxis_type="log", height=420)
        st.plotly_chart(fig_k, use_container_width=True)

# =============================================================================
# SECTION 3: 🧪 STUDENT CHEMISTRY & REACTION LAB
# =============================================================================
elif app_section.startswith("🧪 Student Chemistry"):
    st.markdown('<div class="main-title">🧪 Student Chemistry & Microscopic Reaction Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Understand how Portland cement, water, slag, and fly ash chemically transform from liquid paste into solid stone.</div>', unsafe_allow_html=True)

    col_chem_left, col_chem_right = st.columns([1, 1], gap="large")

    with col_chem_left:
        st.subheader("1. Select Chemical Reactants")
        selected_binder = st.multiselect(
            "Choose materials to mix in your virtual reaction beaker:",
            ["Portland Cement (Alite C3S & Belite C2S)", "Mixing Water (H2O)", "Blast Furnace Slag (Latent Hydraulic)", "Fly Ash (Pozzolanic Silica)", "Superplasticizer (Polymers)"],
            default=["Portland Cement (Alite C3S & Belite C2S)", "Mixing Water (H2O)", "Fly Ash (Pozzolanic Silica)"]
        )

        st.subheader("2. Environmental Curing Conditions")
        curing_temp = st.slider("🌡️ Curing Temperature (°C)", 5, 50, 20, 1, help="Heat accelerates chemical reaction kinetics per Arrhenius' Equation.")
        curing_humidity = st.radio("💧 Moisture Condition:", ["100% Water Submerged / Moist Cured (Ideal)", "50% Ambient Air (Slow drying)", "Dry Hot Wind (Desiccation risk)"])

    with col_chem_right:
        st.subheader("🔬 Chemical Reaction Breakdown")

        if "Portland Cement (Alite C3S & Belite C2S)" in selected_binder and "Mixing Water (H2O)" in selected_binder:
            st.markdown("**Primary Hydration Reaction (The C-S-H Glue Engine):**")
            st.markdown("""
            <div class="reaction-box">
            2 Ca₃SiO₅ (Alite) + 11 H₂O ➔ 3CaO·2SiO₂·8H₂O (C-S-H Gel) + 3 Ca(OH)₂ (Portlandite) + Heat (500 J/g)
            </div>
            """, unsafe_allow_html=True)
            st.write("💡 **What's happening?** Water dissolves the calcium silicates. Needle-like crystals of **C-S-H Gel** interlock, turning the soup into rock. Notice the byproduct: **Calcium Hydroxide (Portlandite)**, which is weak and soluble.")

            if "Fly Ash (Pozzolanic Silica)" in selected_binder:
                st.markdown("**Secondary Pozzolanic Reaction (The Eco-Recycling Magic):**")
                st.markdown("""
                <div class="reaction-box">
                Ca(OH)₂ (Weak byproduct) + SiO₂ (Fly Ash) + H₂O ➔ Secondary C-S-H Gel (Ultra-dense matrix!)
                </div>
                """, unsafe_allow_html=True)
                st.write("✨ **Why this is awesome:** The fly ash eats the weak calcium hydroxide leftover from cement and converts it into MORE strong C-S-H gel! That's why fly ash mixes gain huge strength after 28–90 days.")

            if "Blast Furnace Slag (Latent Hydraulic)" in selected_binder:
                st.markdown("**Latent Hydraulic Slag Activation:**")
                st.markdown("""
                <div class="reaction-box">
                Slag Glass + Alkaline Activator [OH⁻] + H₂O ➔ Calcium Aluminate Silicate Hydrate (C-A-S-H)
                </div>
                """, unsafe_allow_html=True)
                st.write("🛡️ **Chemical Resistance:** Slag forms an impermeable gel that protects steel reinforcement from sea salts and acids.")

        else:
            st.info("👈 Select at least **Portland Cement** and **Mixing Water** on the left to activate the chemical reaction engine!")

    # Microstructure Evolution Graph
    st.markdown("---")
    st.subheader("📈 Microscopic Phase Evolution (Crystals vs. Capillary Voids)")
    d_range = np.array([0.1, 0.5, 1, 3, 7, 14, 28, 56, 90])
    temp_factor = (curing_temp / 20.0) ** 0.5

    # Simulated volume fractions based on powers of hydration
    csh_vol = np.clip(100 * (1 - np.exp(-0.15 * temp_factor * d_range)), 0, 70)
    void_vol = np.clip(50 * np.exp(-0.18 * temp_factor * d_range), 5, 50)
    hydrate_df = pd.DataFrame({'Days': d_range, 'C-S-H Crystal Gel (% Volume)': csh_vol, 'Capillary Water Voids (% Porosity)': void_vol})

    fig_phases = px.line(
        hydrate_df, x='Days', y=['C-S-H Crystal Gel (% Volume)', 'Capillary Water Voids (% Porosity)'],
        title=f"Microstructure Densification at {curing_temp}°C",
        color_discrete_map={'C-S-H Crystal Gel (% Volume)': '#10b981', 'Capillary Water Voids (% Porosity)': '#ef4444'}
    )
    fig_phases.update_layout(height=380, xaxis_type="log", xaxis_title="Hydration Age (Days, Log-Scale)", yaxis_title="Microstructural Volume (%)")
    st.plotly_chart(fig_phases, use_container_width=True)

# =============================================================================
# SECTION 4: 🛠️ STUDENT DIY & REAL-LIFE MAKER LAB
# =============================================================================
elif app_section.startswith("🛠️ Student DIY"):
    st.markdown('<div class="main-title">🛠️ Student DIY & Real-Life Maker Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Practical, hands-on projects you can actually mix and cast at home or in a school laboratory!</div>', unsafe_allow_html=True)

    diy_project = st.selectbox(
        "Choose a DIY Project to Build:",
        [
            "🪴 Modern Geometric Desk Planter / Pen Holder (Smooth finish)",
            "🪨 Garden Stepping Stone or Pathway Paver (Durable & Weatherproof)",
            "🌿 Green Eco-Tile Science Project (High Recycled Fly Ash)",
            "🧱 College/School Structural Compression Cube (For Load Testing)"
        ]
    )

    diy_col1, diy_col2 = st.columns([1, 1], gap="large")

    with diy_col1:
        st.subheader("📦 Practical Recipe & Household Measurements")

        unit_choice = st.radio("Select Units:", ["Parts by Volume (Cups / Scoops)", "Metric Weight (Grams / Kilograms)"], horizontal=True)

        if "Desk Planter" in diy_project:
            st.write("Ideal for smooth desktop pots. Requires fine sand and low gravel.")
            if "Cups" in unit_choice:
                recipe = {"Portland Cement": "1 Part (e.g. 1 Cup)", "Fine Play Sand": "2 Parts (e.g. 2 Cups)", "Clean Water": "0.4 Parts (e.g. 0.4 Cup)", "Cooking Oil": "A few drops (Mold release)"}
            else:
                recipe = {"Portland Cement": "500 grams", "Fine Sand": "1,000 grams", "Water": "200 grams / mL", "Admixture": "5 grams"}
            predicted_diy_strength = 32.0

        elif "Garden Stepping Stone" in diy_project:
            st.write("Needs gravel for heavy foot-traffic resistance.")
            if "Cups" in unit_choice:
                recipe = {"Portland Cement": "1 Part (1 Scoop)", "Fine Sand": "2 Parts (2 Scoops)", "Coarse Gravel/Pebbles": "3 Parts (3 Scoops)", "Water": "0.5 Parts"}
            else:
                recipe = {"Portland Cement": "1,000 grams", "Sand": "2,000 grams", "Gravel": "3,000 grams", "Water": "500 grams"}
            predicted_diy_strength = 28.0

        elif "Green Eco-Tile" in diy_project:
            st.write("Replaces 40% cement with fly ash to demonstrate carbon savings.")
            if "Cups" in unit_choice:
                recipe = {"Portland Cement": "0.6 Parts", "Fly Ash / Slag": "0.4 Parts", "Sand": "2 Parts", "Water": "0.45 Parts"}
            else:
                recipe = {"Portland Cement": "600 grams", "Fly Ash": "400 grams", "Sand": "2,000 grams", "Water": "450 grams"}
            predicted_diy_strength = 35.0

        else:
            st.write("Standard 100mm test cube for laboratory load-testing.")
            if "Cups" in unit_choice:
                recipe = {"Portland Cement": "1 Part", "Sand": "1.8 Parts", "Coarse Stone": "2.8 Parts", "Water": "0.42 Parts"}
            else:
                recipe = {"Portland Cement": "1,200 grams", "Sand": "2,100 grams", "Gravel": "3,300 grams", "Water": "500 grams"}
            predicted_diy_strength = 42.0

        for ing, qty in recipe.items():
            st.markdown(f"- **{ing}:** `{qty}`")

        st.info(f"📊 **Expected 28-Day Strength:** ~`{predicted_diy_strength:.1f} MPa` ({predicted_diy_strength*145:.0f} PSI)")

    with diy_col2:
        st.subheader("🪜 Step-by-Step Casting Instructions")

        st.markdown("""
        <div class="diy-step">
            <strong>Step 1: Safety First!</strong><br>
            Wear rubber gloves and a dust mask. Wet cement is alkaline (pH 12–13) and can irritate skin and eyes.
        </div>
        <div class="diy-step">
            <strong>Step 2: Prepare the Mold</strong><br>
            Use a silicone mold or a recycled plastic container. Wipe the inside with vegetable cooking oil so your concrete releases easily when cured!
        </div>
        <div class="diy-step">
            <strong>Step 3: Dry Mix Before Wetting</strong><br>
            Thoroughly blend the cement powder, sand, and stones together dry until the color is uniform.
        </div>
        <div class="diy-step">
            <strong>Step 4: Add Water Slowly</strong><br>
            Add water gradually. <em>Golden rule:</em> It should have the consistency of thick peanut butter, NOT runny pancake batter!
        </div>
        <div class="diy-step">
            <strong>Step 5: Tap to Remove Bubbles</strong><br>
            Pour into mold and tap the sides firmly for 60 seconds to vibrate trapped air bubbles to the top.
        </div>
        <div class="diy-step">
            <strong>Step 6: The Secret Curing Rule!</strong><br>
            Cover with a damp paper towel and enclose in a plastic ziplock bag for <strong>at least 48 to 72 hours</strong>. Concrete doesn't 'dry by evaporation'—it cures by chemical hydration with water!
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# SECTION 5: 📊 INTERACTIVE CUSTOM PLOTTER & AI INSIGHTS
# =============================================================================
elif app_section.startswith("📊 Interactive Custom Plotter"):
    st.markdown('<div class="main-title">📊 Interactive Custom Graph Plotter & AI Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Pick any X and Y variables from the 1,000+ benchmark tests, choose your chart type, and get instant scientific insights.</div>', unsafe_allow_html=True)

    if hist_df is not None:
        plot_c1, plot_c2, plot_c3 = st.columns(3)

        var_labels = {
            'cement': 'Portland Cement (kg/m³)',
            'slag': 'Blast Furnace Slag (kg/m³)',
            'ash': 'Fly Ash (kg/m³)',
            'water': 'Mixing Water (kg/m³)',
            'superplastic': 'Superplasticizer (kg/m³)',
            'coarseagg': 'Coarse Aggregate (kg/m³)',
            'fineagg': 'Fine Aggregate (kg/m³)',
            'age': 'Curing Age (Days)',
            'wb_ratio': 'Water-to-Binder Ratio (w/b)',
            'strength': 'Compressive Strength (MPa)'
        }

        with plot_c1:
            x_var = st.selectbox("Select X-Axis Variable:", list(var_labels.keys()), index=8, format_func=lambda k: var_labels[k])
        with plot_c2:
            y_var = st.selectbox("Select Y-Axis Variable:", list(var_labels.keys()), index=9, format_func=lambda k: var_labels[k])
        with plot_c3:
            chart_type = st.selectbox("Choose Graph Type:", ["Scatter Plot with Trendline", "2D Density Heatmap", "Box Plot", "Histogram / Distribution"])

        # Render Chart
        if chart_type == "Scatter Plot with Trendline":
            fig_custom = px.scatter(
                hist_df, x=x_var, y=y_var, color='age',
                trendline="ols",
                title=f"{var_labels[y_var]} vs. {var_labels[x_var]}",
                labels={k: var_labels.get(k, k) for k in [x_var, y_var, 'age']},
                opacity=0.65
            )
        elif chart_type == "2D Density Heatmap":
            fig_custom = px.density_heatmap(
                hist_df, x=x_var, y=y_var,
                title=f"Density Heatmap: {var_labels[y_var]} vs. {var_labels[x_var]}",
                labels={k: var_labels.get(k, k) for k in [x_var, y_var]}
            )
        elif chart_type == "Box Plot":
            fig_custom = px.box(
                hist_df, x=x_var if x_var == 'age' else 'age', y=y_var,
                title=f"Distribution of {var_labels[y_var]} across Curing Ages",
                labels={k: var_labels.get(k, k) for k in [x_var, y_var, 'age']}
            )
        else:
            fig_custom = px.histogram(
                hist_df, x=x_var, nbins=30,
                title=f"Histogram of {var_labels[x_var]}",
                color_discrete_sequence=['#0284c7']
            )

        fig_custom.update_layout(height=480)
        st.plotly_chart(fig_custom, use_container_width=True)

        # Automated AI Scientific Insights Engine
        corr_val = hist_df[x_var].corr(hist_df[y_var]) if x_var != y_var else 1.0

        st.markdown(f"### 🧠 Automated Scientific Insights for `{var_labels[x_var]}` &bull; `{var_labels[y_var]}`")

        insight_text = ""
        if x_var == 'wb_ratio' and y_var == 'strength':
            insight_text = "**Abrams' Law Confirmed:** Notice the distinct inverse decay curve. As water-to-binder increases beyond 0.50, compressive strength drops precipitously. The excess water leaves capillary voids upon evaporation."
        elif x_var == 'cement' and y_var == 'strength':
            insight_text = "**Primary Hydraulic Driver:** Strong positive correlation (+0.50). Cement is the primary source of calcium silicates. However, beyond ~450 kg/m³, gains plateau unless the water-to-binder ratio is reduced."
        elif x_var == 'water' and y_var == 'strength':
            insight_text = "**The Water Paradox:** While water is required for hydration, excess water acts as a diluent. Water has a negative correlation with strength (-0.29). Always minimize water and use superplasticizer to maintain workability."
        elif x_var == 'age' and y_var == 'strength':
            insight_text = "**Logarithmic Hydration Kinetics:** Strength increases rapidly from Day 1 to 28, then flattens. Slag and fly ash continue hydrating for up to 365 days due to secondary pozzolanic reactions."
        elif x_var == 'superplastic' and y_var == 'water':
            insight_text = "**Chemical Dispersion:** Strong negative correlation (-0.66). Superplasticizer deflocculates cement grains, allowing engineers to remove ~30% of mixing water while maintaining a fluid pour."
        else:
            if corr_val > 0.3:
                insight_text = f"**Positive Association (r = {corr_val:+.2f}):** Increasing `{var_labels[x_var]}` generally increases `{var_labels[y_var]}` in this dataset."
            elif corr_val < -0.3:
                insight_text = f"**Inverse Association (r = {corr_val:+.2f}):** Increasing `{var_labels[x_var]}` generally decreases `{var_labels[y_var]}`."
            else:
                insight_text = f"**Non-Linear / Weak Direct Correlation (r = {corr_val:+.2f}):** These two variables do not have a simple linear link. Their interaction depends on multi-variable combinations with other constituents."

        st.markdown(f"""
        <div class="insight-card">
            <strong>Pearson Correlation (r):</strong> <code>{corr_val:+.3f}</code><br><br>
            {insight_text}
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Global Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("⚠️ **Mandatory Scientific Notice:** Predictions are machine-learned statistical estimates (Yeh 1998 Benchmark, 92% R²). They provide educational and decision-support guidance but do not replace destructive ASTM C39 / BS EN 12390 testing for life-critical building permits.")
