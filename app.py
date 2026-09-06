"""
app.py
======
Material Behavior Prediction & Intelligent Mix Studio
Two Experiences in One:
1. 🌱 Simple Mode: Everyday language, real-world analogies, project presets, drying timelines, and traffic-light verdicts.
2. 🔬 Advanced Engineering Mode: Precise formulation sliders (kg/m³), Abrams' Law, Hydration Kinetics, Carbon LCA, Batch CSV predictor, and SQLite database.
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
# Page Setup & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Concrete & Material Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 16px;
    }
    .plain-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .verdict-box {
        padding: 14px 18px;
        border-radius: 8px;
        margin: 12px 0;
        font-size: 14px;
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
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# File Paths & Data/Model Caching
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
        df['wb_ratio'] = df['water'] / df['total_binder']
        return df
    return None

model = load_model()
hist_df = load_historical_data()

# -----------------------------------------------------------------------------
# Database Setup
# -----------------------------------------------------------------------------
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

# Carbon LCA Factors
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
# Top-Level Experience Selector
# -----------------------------------------------------------------------------
app_mode = st.radio(
    "Select Your Experience Mode:",
    [
        "🌱 Simple Mode (Everyday Language & Real-World Use Cases)",
        "🔬 Advanced Engineering Mode (Precise Formulas, Charts & Deep Science)"
    ],
    horizontal=True
)

st.markdown("---")

# =============================================================================
# MODE 1: 🌱 SIMPLE MODE (BEGINNER & EVERYDAY USER FRIENDLY)
# =============================================================================
if app_mode.startswith("🌱 Simple"):
    st.markdown('<div class="main-title">🏡 Concrete & Material Strength Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Tell us what you are building or adjust simple ingredients. We will tell you how strong it is in plain English!</div>', unsafe_allow_html=True)

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. What Are You Planning to Build?")
        project_type = st.selectbox(
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

        # Mapping plain use-case to realistic formulation
        if "Backyard Patio" in project_type:
            c_def, s_def, a_def, w_def, sp_def, cg_def, fg_def, age_def = 220.0, 0.0, 50.0, 190.0, 0.0, 1050.0, 780.0, 28
        elif "Residential House" in project_type:
            c_def, s_def, a_def, w_def, sp_def, cg_def, fg_def, age_def = 300.0, 0.0, 0.0, 180.0, 0.0, 1000.0, 750.0, 28
        elif "Garage Floor" in project_type:
            c_def, s_def, a_def, w_def, sp_def, cg_def, fg_def, age_def = 350.0, 80.0, 0.0, 160.0, 6.0, 980.0, 740.0, 28
        elif "Multi-Story" in project_type:
            c_def, s_def, a_def, w_def, sp_def, cg_def, fg_def, age_def = 420.0, 100.0, 0.0, 150.0, 10.0, 940.0, 720.0, 28
        elif "Highway Bridge" in project_type:
            c_def, s_def, a_def, w_def, sp_def, cg_def, fg_def, age_def = 490.0, 90.0, 0.0, 140.0, 16.0, 900.0, 710.0, 28
        elif "Eco-Friendly" in project_type:
            c_def, s_def, a_def, w_def, sp_def, cg_def, fg_def, age_def = 180.0, 140.0, 90.0, 145.0, 10.0, 960.0, 710.0, 56
        else:
            c_def, s_def, a_def, w_def, sp_def, cg_def, fg_def, age_def = 290.0, 50.0, 30.0, 175.0, 5.0, 980.0, 750.0, 28

        st.subheader("2. How Long Has It Dried / Cured?")
        curing_choice = st.select_slider(
            "Select drying time:",
            options=["3 Days (Fresh & Soft)", "7 Days (One Week)", "14 Days (Two Weeks)", "28 Days (Standard Full Strength)", "90 Days (Maximum Age)"],
            value="28 Days (Standard Full Strength)" if age_def == 28 else ("90 Days (Maximum Age)" if age_def == 56 else "28 Days (Standard Full Strength)")
        )

        age_map = {
            "3 Days (Fresh & Soft)": 3,
            "7 Days (One Week)": 7,
            "14 Days (Two Weeks)": 14,
            "28 Days (Standard Full Strength)": 28,
            "90 Days (Maximum Age)": 90
        }
        s_age = age_map[curing_choice]

        st.subheader("3. Fine-Tune Ingredients (Plain English)")
        with st.expander("Adjust Sliders (Optional)", expanded=(project_type.startswith("🛠️"))):
            s_cement = st.slider("🧱 Cement Powder (The glue holding it together)", 100, 550, int(c_def), 10, help="More cement generally increases strength, but costs more.")
            s_water = st.slider("💧 Water Added (Mixing liquid)", 120, 240, int(w_def), 5, help="Rule of thumb: Less water = stronger concrete! Too much water makes it soupy and weak.")
            s_sand_stone = st.slider("🪨 Sand & Crushed Stones (The solid rock skeleton)", 1400, 2000, int(cg_def + fg_def), 20, help="Stones and sand take up bulk volume.")
            s_eco_binder = st.slider("🌿 Recycled Slag & Ash (Eco-friendly cement replacements)", 0, 300, int(s_def + a_def), 10, help="Recycled industrial minerals that reduce pollution.")

            # Distribute sand vs stone and slag vs ash proportionately
            s_slag = float(s_eco_binder * 0.6)
            s_ash = float(s_eco_binder * 0.4)
            s_coarse = float(s_sand_stone * 0.58)
            s_fine = float(s_sand_stone * 0.42)
            s_superplastic = sp_def

    # Calculations for Simple Mode
    sim_pred = predict_strength(s_cement, s_slag, s_ash, s_water, s_superplastic, s_coarse, s_fine, s_age)
    psi_equiv = sim_pred * 145.038
    sim_carbon = calculate_carbon_footprint(s_cement, s_slag, s_ash, s_water, s_superplastic, s_coarse, s_fine)

    with col_output:
        st.subheader("📋 Your Result in Everyday Terms")

        # Plain language classification
        if sim_pred < 20.0:
            rating_title = "Light Duty (Garden & Pathway Grade)"
            rating_desc = "Good for simple decorative paths, garden stepping stones, or non-load-bearing fill."
            box_bg = "#fef3c7"
            box_border = "#f59e0b"
            box_color = "#92400e"
            icon = "🌱"
        elif 20.0 <= sim_pred < 35.0:
            rating_title = "Solid Structural Grade (Standard Home Quality)"
            rating_desc = "Strong enough for residential house slabs, standard garage floors, and outdoor driveways."
            box_bg = "#e0f2fe"
            box_border = "#0284c7"
            box_color = "#075985"
            icon = "🏠"
        elif 35.0 <= sim_pred < 50.0:
            rating_title = "Heavy-Duty Commercial Grade (Very Strong)"
            rating_desc = "Easily supports heavy delivery trucks, parking garages, and multi-story commercial buildings."
            box_bg = "#dcfce7"
            box_border = "#10b981"
            box_color = "#065f46"
            icon = "🚛"
        else:
            rating_title = "Ultra High-Performance (Extreme Strength)"
            rating_desc = "Engineered for skyscrapers, highway bridges, and heavy industrial load-bearing piers."
            box_bg = "#f3e8ff"
            box_border = "#8b5cf6"
            box_color = "#5b21b6"
            icon = "🏙️"

        # Verdict Card
        st.markdown(f"""
        <div style="background:{box_bg}; border: 2px solid {box_border}; border-radius: 10px; padding: 16px; color: {box_color};">
            <h3 style="margin:0 0 6px 0; color: {box_color};">{icon} Strength Verdict: {rating_title}</h3>
            <p style="margin:0; font-size: 14px;">{rating_desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Real World Analogy Box
        elephant_weight = 4000  # kg (~8,800 lbs)
        num_cars = int(psi_equiv / 1500)
        st.markdown(f"""
        <div class="analogy-box">
            <strong>🐘 What does this mean in real life?</strong><br>
            This mix can withstand roughly <strong>{psi_equiv:,.0f} pounds of force per square inch (PSI)</strong>.<br>
            That means a small tile (1 foot &times; 1 foot) could hold the weight of roughly <strong>{max(1, num_cars)} full-size SUVs</strong> stacked on top without crushing!
        </div>
        """, unsafe_allow_html=True)

        # Can you use it for checklist
        st.markdown("**Suitable for your project?**")
        chk1 = "✅ Yes" if sim_pred >= 15 else "❌ Too weak"
        chk2 = "✅ Yes" if sim_pred >= 25 else "❌ Needs more strength"
        chk3 = "✅ Yes" if sim_pred >= 35 else "⚠️ Commercial mix recommended"
        chk4 = "✅ Yes" if sim_pred >= 45 else "❌ Not suitable for skyscrapers"

        c1, c2 = st.columns(2)
        c1.write(f"{chk1} — Garden Patios & Sidewalks")
        c1.write(f"{chk2} — House Foundations & Driveways")
        c2.write(f"{chk3} — Heavy Commercial & Parking Lots")
        c2.write(f"{chk4} — High-Rise Bridges & Skyscraper Piers")

        # Drying Timeline
        st.markdown("---")
        st.markdown("**⏱️ Safe Activity Timeline (When is it ready?):**")
        s3 = predict_strength(s_cement, s_slag, s_ash, s_water, s_superplastic, s_coarse, s_fine, 3)
        s7 = predict_strength(s_cement, s_slag, s_ash, s_water, s_superplastic, s_coarse, s_fine, 7)
        s28 = predict_strength(s_cement, s_slag, s_ash, s_water, s_superplastic, s_coarse, s_fine, 28)

        t1, t2, t3 = st.columns(3)
        t1.metric("Day 3", f"{s3:.1f} MPa", "🚶 Safe to walk on")
        t2.metric("Day 7", f"{s7:.1f} MPa", "🚗 Light vehicle safe")
        t3.metric("Day 28", f"{s28:.1f} MPa", "🏆 Full design strength")

        # Eco Card
        if (s_slag + s_ash) > 50:
            st.success(f"🌿 **Eco-Friendly Mix:** Saves approx. {max(0, int(350 - sim_carbon))} kg of carbon per m³ compared to pure cement!")
        else:
            st.info("💡 **Eco-Tip:** Adding recycled slag or fly ash cuts carbon emissions by up to 40% while keeping the concrete just as strong!")

    # -------------------------------------------------------------------------
    # Expandable "Click for Deep Science" in Simple Mode
    # -------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("🔬 Curious about the Deep Science? (Click to view exact numbers & charts)", expanded=False):
        st.markdown("### Exact Laboratory Figures & Physical Laws")
        sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
        sc_col1.metric("Exact Strength", f"{sim_pred:.2f} MPa")
        sc_col2.metric("ASTM C39 Margin", f"±3.45 MPa")
        sc_col3.metric("Water/Binder Ratio", f"{s_water/(s_cement+s_slag+s_ash):.2f}")
        sc_col4.metric("Embodied CO₂", f"{sim_carbon:.1f} kg/m³")

        # Mini kinetics chart
        sub_days = [1, 3, 7, 14, 28, 56, 90, 180, 365]
        sub_strengths = [predict_strength(s_cement, s_slag, s_ash, s_water, s_superplastic, s_coarse, s_fine, d) for d in sub_days]
        fig_sub = px.line(x=sub_days, y=sub_strengths, labels={'x': 'Curing Days', 'y': 'Strength (MPa)'}, title="Strength Growth Over 365 Days")
        fig_sub.add_scatter(x=[s_age], y=[sim_pred], mode='markers', name=f'Your Day ({s_age}d)', marker=dict(size=12, color='red'))
        st.plotly_chart(fig_sub, use_container_width=True)


# =============================================================================
# MODE 2: 🔬 ADVANCED ENGINEERING MODE (CIVIL & MATERIALS SCIENTISTS)
# =============================================================================
else:
    st.markdown('<div class="main-title">🔬 Advanced Material Behavior & Mix Design Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Complete formulation design, ASTM error bounds, Abrams\' Law analytics, and SQLite persistence.</div>', unsafe_allow_html=True)

    # Sidebar: Engineering sliders
    st.sidebar.title("🎛️ Engineering Controls")
    adv_preset = st.sidebar.selectbox(
        "Select Mix Template / Preset:",
        [
            "Standard Residential (C25/30) - 28d",
            "Eco-Green Low-Carbon (High Slag + Ash) - 56d",
            "Rapid Early-Strength Repair (Highway/Airport) - 3d",
            "High-Performance Skyscraper Mix (C60/75) - 28d",
            "Lean Mass Concrete (C15/20) - 28d",
            "Custom Formulation"
        ]
    )

    if adv_preset == "Standard Residential (C25/30) - 28d":
        p_c, p_s, p_a, p_w, p_sp, p_cg, p_fg, p_age = 300.0, 0.0, 0.0, 180.0, 0.0, 1000.0, 750.0, 28
    elif adv_preset == "Eco-Green Low-Carbon (High Slag + Ash) - 56d":
        p_c, p_s, p_a, p_w, p_sp, p_cg, p_fg, p_age = 200.0, 130.0, 90.0, 150.0, 10.0, 950.0, 720.0, 56
    elif adv_preset == "Rapid Early-Strength Repair (Highway/Airport) - 3d":
        p_c, p_s, p_a, p_w, p_sp, p_cg, p_fg, p_age = 460.0, 40.0, 0.0, 140.0, 16.0, 910.0, 690.0, 3
    elif adv_preset == "High-Performance Skyscraper Mix (C60/75) - 28d":
        p_c, p_s, p_a, p_w, p_sp, p_cg, p_fg, p_age = 480.0, 100.0, 0.0, 150.0, 18.0, 880.0, 740.0, 28
    elif adv_preset == "Lean Mass Concrete (C15/20) - 28d":
        p_c, p_s, p_a, p_w, p_sp, p_cg, p_fg, p_age = 180.0, 0.0, 60.0, 195.0, 0.0, 1050.0, 800.0, 28
    else:
        p_c, p_s, p_a, p_w, p_sp, p_cg, p_fg, p_age = 280.0, 80.0, 50.0, 170.0, 7.0, 970.0, 760.0, 28

    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Hydraulic Binders (kg/m³)")
    cement = st.sidebar.slider("Portland Cement", 100.0, 550.0, float(p_c), 5.0)
    slag = st.sidebar.slider("Blast Furnace Slag", 0.0, 360.0, float(p_s), 5.0)
    ash = st.sidebar.slider("Fly Ash", 0.0, 200.0, float(p_a), 5.0)

    st.sidebar.subheader("2. Water & Admixtures (kg/m³)")
    water = st.sidebar.slider("Mixing Water", 120.0, 250.0, float(p_w), 1.0)
    superplastic = st.sidebar.slider("Superplasticizer", 0.0, 35.0, float(p_sp), 0.5)

    st.sidebar.subheader("3. Aggregates (kg/m³)")
    coarseagg = st.sidebar.slider("Coarse Aggregate (Gravel)", 800.0, 1150.0, float(p_cg), 10.0)
    fineagg = st.sidebar.slider("Fine Aggregate (Sand)", 590.0, 1000.0, float(p_fg), 10.0)

    st.sidebar.subheader("4. Curing Duration (Days)")
    age = st.sidebar.slider("Curing Age", 1, 365, int(p_age), 1)

    batch_name = st.sidebar.text_input("Mix Name / Batch ID", value="Lab-Trial-01")

    # Metrics
    tot_binder = cement + slag + ash
    wb = water / tot_binder if tot_binder > 0 else 0.0
    tot_density = cement + slag + ash + water + superplastic + coarseagg + fineagg
    co2_val = calculate_carbon_footprint(cement, slag, ash, water, superplastic, coarseagg, fineagg)
    pred_val = predict_strength(cement, slag, ash, water, superplastic, coarseagg, fineagg, age)

    # Top KPI Ribbon
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted Strength (f'c)", f"{pred_val:.2f} MPa", f"{pred_val-35.82:+.1f} vs Avg")
    k2.metric("w/b Ratio", f"{wb:.2f}", "Denser / Stronger" if wb < 0.45 else "Porous", delta_color="inverse" if wb < 0.45 else "normal")
    k3.metric("Embodied CO₂", f"{co2_val:.1f} kg CO₂e/m³", f"{(co2_val-350):+.0f} vs OPC Baseline", delta_color="inverse")
    k4.metric("Density", f"{tot_density:.0f} kg/m³", "Normal Weight" if 2200 <= tot_density <= 2550 else "Check Envelope")

    if st.button("💾 Save Mix Ticket to SQLite Database"):
        save_mix_to_db(batch_name, cement, slag, ash, water, superplastic, coarseagg, fineagg, age, round(pred_val, 2), round(wb, 2), round(co2_val, 1), "Advanced Mix")
        st.toast(f"Saved {batch_name} to database!", icon="✅")

    # Tabs
    t_kinetics, t_sens, t_abrams, t_carbon, t_cmp, t_csv, t_db = st.tabs([
        "📈 Hydration Kinetics", "⚖️ Sensitivity Sweep", "🔬 Abrams' Law",
        "🌿 Carbon & SCMs", "⚔️ Mix Comparator", "📁 Batch CSV", "🗄️ Database"
    ])

    with t_kinetics:
        st.subheader("Hydration Kinetics Curve (Days 1 to 365)")
        sim_days = [1, 3, 7, 14, 21, 28, 56, 90, 120, 180, 270, 365]
        sim_strengths = [predict_strength(cement, slag, ash, water, superplastic, coarseagg, fineagg, d) for d in sim_days]

        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=sim_days, y=sim_strengths, mode='lines+markers', name='Progression', line=dict(color='#0284c7', width=3)))
        fig_k.add_trace(go.Scatter(x=[age], y=[pred_val], mode='markers', name=f'Current ({age}d)', marker=dict(size=14, color='red', symbol='star')))
        fig_k.update_layout(xaxis_title="Curing Age (Days, Log-scale)", yaxis_title="Compressive Strength (MPa)", xaxis_type="log", height=430)
        st.plotly_chart(fig_k, use_container_width=True)

    with t_sens:
        st.subheader("Parameter Sensitivity Sweep")
        s_feat = st.selectbox("Feature to sweep:", ["water", "cement", "superplastic", "slag", "ash", "age"])
        sweep_range = {
            "water": (120, 250), "cement": (100, 550), "superplastic": (0, 35),
            "slag": (0, 360), "ash": (0, 200), "age": (1, 180)
        }[s_feat]
        sw_x = np.linspace(sweep_range[0], sweep_range[1], 40)
        sw_y = []
        for v in sw_x:
            sw_y.append(predict_strength(
                v if s_feat == 'cement' else cement,
                v if s_feat == 'slag' else slag,
                v if s_feat == 'ash' else ash,
                v if s_feat == 'water' else water,
                v if s_feat == 'superplastic' else superplastic,
                coarseagg, fineagg,
                int(v) if s_feat == 'age' else age
            ))
        fig_s = px.line(x=sw_x, y=sw_y, labels={'x': s_feat, 'y': 'Strength (MPa)'}, title=f"Sensitivity to {s_feat}")
        st.plotly_chart(fig_s, use_container_width=True)

    with t_abrams:
        st.subheader("Abrams' Law Validation Scatter")
        if hist_df is not None:
            fig_a = px.scatter(hist_df, x='wb_ratio', y='strength', color='age', opacity=0.5, title="Benchmark Tests vs Candidate Mix")
            fig_a.add_trace(go.Scatter(x=[wb], y=[pred_val], mode='markers', name='Your Mix', marker=dict(size=16, color='red', symbol='star')))
            st.plotly_chart(fig_a, use_container_width=True)

    with t_carbon:
        st.subheader("Embodied Carbon Breakdown")
        c_dict = {'Cement': cement*0.86, 'Slag': slag*0.07, 'Ash': ash*0.015, 'Admixture': superplastic*0.25, 'Aggregates': (coarseagg+fineagg)*0.005}
        fig_c = px.pie(names=list(c_dict.keys()), values=list(c_dict.values()), hole=0.4, title="Carbon Footprint by Constituent")
        st.plotly_chart(fig_c, use_container_width=True)

    with t_cmp:
        st.subheader("Side-by-Side Mix Comparator")
        b_c, b_w = 400.0, 180.0
        pred_base = predict_strength(b_c, 0, 0, b_w, 0, 1000, 750, 28)
        c_df = pd.DataFrame({'Mix': ['Your Mix', 'OPC Baseline'], 'Strength': [pred_val, pred_base]})
        st.plotly_chart(px.bar(c_df, x='Mix', y='Strength', color='Mix', title="Strength Comparison (MPa)"), use_container_width=True)

    with t_csv:
        st.subheader("Batch File Predictor")
        u_file = st.file_uploader("Upload CSV", type=["csv"])
        if u_file is not None:
            b_df = pd.read_csv(u_file)
            req = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']
            if all(r in b_df.columns for r in req):
                b_df['predicted_strength'] = np.round(model.predict(b_df[req]), 2)
                st.dataframe(b_df)
                st.download_button("Download CSV", data=b_df.to_csv(index=False).encode('utf-8'), file_name="predictions.csv")

    with t_db:
        st.subheader("Database Archive")
        trials = get_all_trials()
        if len(trials) > 0:
            st.dataframe(trials)
            st.download_button("Export Archive CSV", data=trials.to_csv(index=False).encode('utf-8'), file_name="trials_archive.csv")
        else:
            st.info("No mixes logged yet.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("⚠️ **Scientific Notice:** Predictions are machine-learned statistical estimates based on empirical concrete data (Yeh 1998). They do not replace ASTM C39 physical laboratory testing for life-critical structural designs.")
