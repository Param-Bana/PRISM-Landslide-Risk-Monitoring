
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "prism_landslide_rf_pipeline_updated.joblib"

model = joblib.load(MODEL_PATH)

STATES = [
    "Arunachal Pradesh", "Assam", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Sikkim", "Tripura"
]

SOIL_TYPES = [
    "Alluvial", "Clay", "Sandy", "Loamy", "Silty",
    "Lateritic", "Rocky / Shallow Soil", "Mountain Soil",
    "Other / Unknown"
]

LAND_COVERS = [
    "Dense Forest", "Open Forest", "Grassland", "Cropland",
    "Built-up", "Barren Land", "Shrubland", "Water Body",
    "Mountain Terrain"
]

st.set_page_config(page_title="PRISM Landslide Risk Prototype", layout="wide")
st.title("PRISM — AI Landslide Risk Monitoring Prototype")
st.caption("Random Forest susceptibility prototype using terrain, rainfall, vegetation, soil, land cover and proximity features.")

st.info(
    "Mountain Terrain is included as a user-facing terrain category. "
    "For an operational model, mountain terrain should additionally be represented "
    "through elevation, slope, relief and geomorphology rather than treated as a conventional land-cover class."
)

with st.form("risk_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        state = st.selectbox("State", STATES)
        soil_type = st.selectbox("Soil Type", SOIL_TYPES)
        land_cover = st.selectbox("Land Cover / Terrain", LAND_COVERS)
        elevation = st.number_input("Elevation (m)", min_value=0.0, max_value=9000.0, value=1200.0)
        slope = st.number_input("Slope (degrees)", min_value=0.0, max_value=90.0, value=30.0)

    with c2:
        aspect = st.number_input("Aspect (degrees)", min_value=0.0, max_value=360.0, value=180.0)
        curvature = st.number_input("Curvature", value=0.0)
        ndvi = st.number_input("NDVI", min_value=-1.0, max_value=1.0, value=0.5)
        rain24 = st.number_input("Rainfall — 24 h (mm)", min_value=0.0, value=50.0)
        rain72 = st.number_input("Rainfall — 72 h (mm)", min_value=0.0, value=100.0)

    with c3:
        rain7 = st.number_input("Rainfall — 7 d (mm)", min_value=0.0, value=150.0)
        river = st.number_input("Distance to River (m)", min_value=0.0, value=500.0)
        submitted = st.form_submit_button("Analyze Susceptibility", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "elevation_m": elevation,
        "slope_deg": slope,
        "aspect_deg": aspect,
        "curvature": curvature,
        "rainfall_24h_mm": rain24,
        "rainfall_72h_mm": rain72,
        "rainfall_7d_mm": rain7,
        "ndvi": ndvi,
        "distance_to_river_m": river,
        "soil_type": soil_type,
        "land_cover": land_cover,
        "state": state
    }])

    score = float(model.predict_proba(row)[0, 1])

    if score < 0.25:
        level, color = "GREEN — Low", "green"
    elif score < 0.50:
        level, color = "YELLOW — Moderate", "orange"
    elif score < 0.75:
        level, color = "ORANGE — High", "orange"
    else:
        level, color = "RED — Very High", "red"

    st.subheader("PRISM Susceptibility Result")
    st.metric("Model Score", f"{score:.1%}")
    st.markdown(f"### :{color}[{level}]")

    st.caption(
        "This is a prototype model score, not a calibrated operational probability "
        "or an official early-warning threshold."
    )
