# app.py
from datetime import date, time
import joblib
import pandas as pd
import streamlit as st
from pipeline import (
    AIRPORT_COORDS,
    FEATURE_COLS,
    fetch_airport_weather,
    haversine_distance_miles,
)

st.set_page_config(page_title="Will Your Flight Be Late?", page_icon="✈️", layout="centered")


@st.cache_resource
def load_artifact(path: str = "models/lgbm_flight_model.pkl"):
    """Caches the model in memory so it is only loaded once across all users."""
    return joblib.load(path)


st.title("✈️ Will Your Flight Be Late?")
st.markdown(
    "Check the probability of your US flight arriving **15 or more minutes late** "
    "based on US Bureau of Transportation Statistics (BTS) records and live Open-Meteo weather."
)

try:
    artifact = load_artifact()
    model = artifact["model"]
    cat_mappings = artifact["categories"]
except FileNotFoundError:
    st.error("Model file `models/lgbm_flight_model.pkl` not found. Run `python train.py` first.")
    st.stop()

# User Input Form
with st.form("flight_check_form"):
    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox(
            "Airline (IATA Code)",
            options=cat_mappings.get("Reporting_Airline", ["AA", "DL", "UA", "WN", "B6"]),
        )
        origin = st.selectbox("Origin Airport", options=sorted(AIRPORT_COORDS.keys()), index=0)
        flight_date = st.date_input("Flight Date", value=date.today())

    with col2:
        dest = st.selectbox("Destination Airport", options=sorted(AIRPORT_COORDS.keys()), index=5)
        dep_time = st.time_input("Scheduled Departure Time", value=time(14, 30))

    submitted = st.form_submit_button("Check Flight Delay Risk", use_container_width=True)

if submitted:
    if origin == dest:
        st.warning("Origin and Destination airports must be different.")
        st.stop()

    dep_hour = dep_time.hour
    distance = haversine_distance_miles(origin, dest)
    weather = fetch_airport_weather(origin, flight_date.isoformat(), dep_hour)

    # Construct single-row DataFrame matching FEATURE_COLS order
    input_df = pd.DataFrame(
        [
            {
                "Month": int(flight_date.month),
                "DayOfWeek": int(flight_date.isoweekday()),  # 1=Mon, 7=Sun (matches BTS schema)
                "Reporting_Airline": airline,
                "Origin": origin,
                "Dest": dest,
                "dep_hour": int(dep_hour),
                "Distance": float(distance),
                "temp_c": weather["temp_c"],
                "wind_speed_kmh": weather["wind_speed_kmh"],
                "precipitation_mm": weather["precipitation_mm"],
            }
        ]
    )[FEATURE_COLS]

    # Enforce exact categorical levels from training so LightGBM never mismatches codes
    for col, categories in cat_mappings.items():
        input_df[col] = pd.Categorical(input_df[col], categories=categories)

    # Predict probability of ArrDel15 == 1
    delay_prob = float(model.predict_proba(input_df)[0, 1])
    pct = round(delay_prob * 100, 1)

    st.divider()
    if delay_prob >= 0.50:
        st.error(f"### ⚠️ High Delay Risk: {pct}% chance of arriving 15+ mins late")
    elif delay_prob >= 0.35:
        st.warning(f"### 🟡 Moderate Delay Risk: {pct}% chance of arriving 15+ mins late")
    else:
        st.success(f"### 🟢 Low Delay Risk: {pct}% chance of arriving 15+ mins late")

    # Show transparent inputs used for prediction
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Route Distance", f"{int(distance)} mi")
    m2.metric("Origin Temp", f"{weather['temp_c']} °C")
    m3.metric("Wind Speed", f"{weather['wind_speed_kmh']} km/h")
    m4.metric("Precipitation", f"{weather['precipitation_mm']} mm")