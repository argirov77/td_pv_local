import logging
import os
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "0d8ab490dd584b518ee91719252111")
FORECAST_API_URL = os.getenv(
    "WEATHER_API_URL", "https://api.weatherapi.com/v1/forecast.json"
)
HISTORY_API_URL = os.getenv(
    "WEATHER_HISTORY_API_URL", "https://api.weatherapi.com/v1/history.json"
)


def _resample_to_quarter_hour(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    numeric_cols = [c for c in df.columns if c != "time"]
    df = df.resample("15min").interpolate()
    df = df[numeric_cols].reset_index()
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M")
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt_idx = pd.to_datetime(df["time"])
    df["hour_local"] = dt_idx.dt.hour + dt_idx.dt.minute / 60.0
    df["dayofyear"] = dt_idx.dt.dayofyear

    df["hour_sin"] = np.sin(df["hour_local"] * 2 * np.pi / 24)
    df["hour_cos"] = np.cos(df["hour_local"] * 2 * np.pi / 24)
    df["doy_sin"] = np.sin(df["dayofyear"] * 2 * np.pi / 365)
    df["doy_cos"] = np.cos(df["dayofyear"] * 2 * np.pi / 365)
    return df


def fetch_weather_forecast(latitude: float, longitude: float, target_date: date) -> Optional[List[dict]]:
    if not WEATHER_API_KEY:
        logger.error("WEATHER_API_KEY is not set")
        return None

    api_url = FORECAST_API_URL if target_date >= date.today() else HISTORY_API_URL

    params = {
        "key": WEATHER_API_KEY,
        "q": f"{latitude},{longitude}",
        "dt": target_date.strftime("%Y-%m-%d"),
        "aqi": "no",
        "alerts": "no",
    }

    try:
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Weather API request failed: %s", exc)
        return None

    payload = response.json()
    forecast_days = payload.get("forecast", {}).get("forecastday", [])
    if not forecast_days:
        logger.error("Weather API response missing forecast data")
        return None

    day_str = target_date.strftime("%Y-%m-%d")
    day_data = next((d for d in forecast_days if str(d.get("date")) == day_str), None)
    if not day_data:
        logger.error("No forecast day matching %s in response", day_str)
        return None

    hours = day_data.get("hour", [])
    if not hours:
        logger.error("No hourly data found in Weather API response")
        return None

    records = []
    for hour in hours:
        records.append(
            {
                "time": hour.get("time"),
                "temp_c": hour.get("temp_c"),
                "cloud": hour.get("cloud"),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return None

    df = _resample_to_quarter_hour(df)
    df = _add_time_features(df)
    return df.to_dict(orient="records")
