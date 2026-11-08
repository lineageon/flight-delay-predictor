# pipeline.py
import requests
import pandas as pd
import numpy as np

# Top US Airport coordinates (Lat, Lon) and Distance Lookup (Miles)
# Easily expandable; keeps our app self-contained and fast
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277),
    "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737),
    "ORD": (41.9742, -87.9073),
    "LAX": (33.9416, -118.4085),
    "JFK": (40.6413, -73.7781),
    "LAS": (36.0840, -115.1537),
    "MCO": (28.4312, -81.3081),
    "MIA": (25.7959, -80.2870),
    "SEA": (47.4502, -122.3088),
    "SFO": (37.6213, -122.3790),
    "EWR": (40.6895, -74.1745),
    "BOS": (42.3656, -71.0096),
    "PHX": (33.4373, -112.0078),
}

CATEGORICAL_COLS = ["Reporting_Airline", "Origin", "Dest"]
FEATURE_COLS = [
    "Month",
    "DayOfWeek",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "dep_hour",
    "Distance",
    "temp_c",
    "wind_speed_kmh",
    "precipitation_mm",
]


def convert_crs_dep_time_to_hour(crs_time_series: pd.Series) -> pd.Series:
    """
    BTS stores CRSDepTime as an integer in HHMM format (e.g., 830 -> 8:30 AM, 1745 -> 5:45 PM).
    Integer division by 100 extracts the hour (0 to 23). Hour 24 is normalized to 0.
    """
    hours = (crs_time_series.fillna(0).astype(int) // 100).clip(0, 24)
    return hours.replace(24, 0).astype("int8")


def haversine_distance_miles(origin: str, dest: str) -> float:
    """Calculates great-circle distance in miles between two airports using pure math."""
    if origin not in AIRPORT_COORDS or dest not in AIRPORT_COORDS:
        return 800.0  # Safe fallback average domestic distance
    lat1, lon1 = np.radians(AIRPORT_COORDS[origin])
    lat2, lon2 = np.radians(AIRPORT_COORDS[dest])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float(round(3958.8 * c, 1))  # Earth radius in miles: 3958.8


def fetch_airport_weather(airport_code: str, date_str: str, dep_hour: int) -> dict:
    """
    Fetches hourly temperature (C), wind speed (km/h), and precipitation (mm)
    for a specific airport, date (YYYY-MM-DD), and hour (0-23) using Open-Meteo.
    Includes a resilient fallback so network hiccups never crash the app.
    """
    default_weather = {"temp_c": 18.0, "wind_speed_kmh": 12.0, "precipitation_mm": 0.0}
    if airport_code not in AIRPORT_COORDS:
        return default_weather

    lat, lon = AIRPORT_COORDS[airport_code]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "auto",
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        hourly = response.json().get("hourly", {})
        idx = min(max(int(dep_hour), 0), 23)
        return {
            "temp_c": float(hourly["temperature_2m"][idx]),
            "wind_speed_kmh": float(hourly["wind_speed_10m"][idx]),
            "precipitation_mm": float(hourly["precipitation"][idx]),
        }
    except Exception:
        # Fallback if offline or date is out of forecast range
        return default_weather