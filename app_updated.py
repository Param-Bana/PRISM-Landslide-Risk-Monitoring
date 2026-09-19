import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "prism_landslide_rf_pipeline_no_road.joblib"
model = joblib.load(MODEL_PATH)
NUMERIC = [
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "curvature",
    "rainfall_24h_mm",
    "rainfall_72h_mm",
    "rainfall_7d_mm",
    "ndvi",
    "distance_to_river_m",
]

CATEGORICAL = [
    "soil_type",
    "land_cover",
    "state",
]

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

st.set_page_config(page_title="Aashray Landslide Risk", layout="wide")
st.title("Aashray — AI Landslide Risk Monitoring Prototype")

with st.form("risk_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        state = st.selectbox("State", STATES)
        soil_type = st.selectbox("Soil Type", SOIL_TYPES)
        land_cover = st.selectbox("Land Cover / Terrain", LAND_COVERS)
        elevation = st.number_input("Elevation (m)", 0.0, 9000.0, 1200.0)
        slope = st.number_input("Slope (degrees)", 0.0, 90.0, 30.0)

    with c2:
        aspect = st.number_input("Aspect (degrees)", 0.0, 360.0, 180.0)
        curvature = st.number_input("Curvature", value=0.0)
        ndvi = st.number_input("NDVI", -1.0, 1.0, 0.5)
        rain24 = st.number_input("Rainfall — 24 h (mm)", 0.0, value=50.0)
        rain72 = st.number_input("Rainfall — 72 h (mm)", 0.0, value=100.0)

    with c3:
        rain7 = st.number_input("Rainfall — 7 d (mm)", 0.0, value=150.0)
        river = st.number_input("Distance to River (m)", 0.0, value=500.0)
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

    # Safety check: road distance is NOT part of this model.
    expected = set(NUMERIC + CATEGORICAL)

    if set(row.columns) != expected:
        st.error("Input/model feature mismatch.")
        st.stop()

    # Model prediction
    score = float(model.predict_proba(row)[0, 1])

    # Risk classification
    if score < 0.25:
        level = "GREEN — Low"
        risk_color = "#22c55e"
    elif score < 0.50:
        level = "YELLOW — Moderate"
        risk_color = "#eab308"
    elif score < 0.75:
        level = "ORANGE — High"
        risk_color = "#f97316"
    else:
        level = "RED — Very High"
        risk_color = "#ef4444"

    st.subheader("Aashray Landslide Susceptibility Result")

    # Colored risk box
    st.markdown(
        f"""
        <div style="
            background-color: {risk_color};
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 30px;
            font-weight: 700;
            margin: 15px 0;
        ">
            {level}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric("Model Score", f"{score:.1%}")

    st.caption(
        "Prototype model score; not a calibrated operational warning probability."
    )