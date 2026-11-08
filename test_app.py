# test_app.py
import pandas as pd
import lightgbm as lgb
from pipeline import (
    convert_crs_dep_time_to_hour,
    haversine_distance_miles,
    fetch_airport_weather,
    FEATURE_COLS,
    CATEGORICAL_COLS,
)


def test_crs_dep_time_conversion():
    raw_times = pd.Series([5, 830, 1445, 2359, 2400, None])
    hours = convert_crs_dep_time_to_hour(raw_times).tolist()
    assert hours == [0, 8, 14, 23, 0, 0]


def test_haversine_distance():
    dist = haversine_distance_miles("LAX", "JFK")
    # LAX to JFK great-circle distance is roughly 2,475 miles
    assert 2400.0 <= dist <= 2550.0


def test_weather_fallback_on_unknown_airport():
    weather = fetch_airport_weather("ZZZ_INVALID", "2026-11-15", 12)
    assert set(weather.keys()) == {"temp_c", "wind_speed_kmh", "precipitation_mm"}
    assert isinstance(weather["temp_c"], float)


def test_lightgbm_schema_and_probability_bounds():
    # Create a tiny synthetic 20-row dataset to verify model pipeline without needing the 2GB CSV
    sample_df = pd.DataFrame(
        {
            "Month": [6, 12] * 10,
            "DayOfWeek": [1, 5] * 10,
            "Reporting_Airline": pd.Categorical(["AA", "DL"] * 10),
            "Origin": pd.Categorical(["ATL", "ORD"] * 10),
            "Dest": pd.Categorical(["JFK", "LAX"] * 10),
            "dep_hour": [8, 19] * 10,
            "Distance": [760.0, 1740.0] * 10,
            "temp_c": [25.0, -2.0] * 10,
            "wind_speed_kmh": [10.0, 35.0] * 10,
            "precipitation_mm": [0.0, 8.5] * 10,
        }
    )[FEATURE_COLS]
    target = pd.Series([0, 1] * 10)

    clf = lgb.LGBMClassifier(n_estimators=10, min_child_samples=2, random_state=42, verbose=-1)
    clf.fit(sample_df, target, categorical_feature=CATEGORICAL_COLS)

    prob = float(clf.predict_proba(sample_df.iloc[[0]])[0, 1])
    assert 0.0 <= prob <= 1.0