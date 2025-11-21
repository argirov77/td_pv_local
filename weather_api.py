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
FORECAST_API_URL = os.getenv("WEATHER_API_URL", "https://api.weatherapi.com/v1/forecast.json")
HISTORY_API_URL = os.getenv("WEATHER_HISTORY_API_URL", "https://api.weatherapi.com/v1/history.json")


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


def interpolate_15min(values, date_str):
    times = pd.date_range(f"{date_str} 00:00", f"{date_str} 23:00", freq="1H")
    series = pd.Series(values, index=times)
    new_times = pd.date_range(f"{date_str} 00:00", f"{date_str} 23:45", freq="15T")
    interpolated = series.reindex(new_times).interpolate("linear")
    return [t.strftime("%Y-%m-%d %H:%M") for t in new_times], interpolated.tolist()


def get_forecast_by_coords(lat: float, lon: float, forecast_date: date):
    query = f"{lat},{lon}"
    params = {
        "key": WEATHER_API_KEY,
        "q": query,
        "dt": forecast_date.strftime("%Y-%m-%d"),
    }

    try:
        resp = requests.get(FORECAST_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        hours = data["forecast"]["forecastday"][0]["hour"]
        temp_c = [h.get("temp_c") for h in hours]
        cloud = [h.get("cloud") for h in hours]

        date_str = forecast_date.strftime("%Y-%m-%d")
        time_15min, temp_c_15min = interpolate_15min(temp_c, date_str)
        _, cloud_15min = interpolate_15min(cloud, date_str)

        temp_c_15min = [round(val, 1) if val is not None else None for val in temp_c_15min]
        cloud_15min = [int(round(val)) if val is not None else None for val in cloud_15min]

        tz = data.get("location", {}).get("tz_id")

        return {
            "date": date_str,
            "location": data.get("location", {}).get("name"),
            "lat": lat,
            "lon": lon,
            "tz": tz,
            "time": time_15min,
            "temp_c": temp_c_15min,
            "cloud": cloud_15min,
        }
    except Exception as exc:
        logger.error("WeatherAPI forecast error: %s", exc)
        return None


def extract_weather_history(lat: float, lon: float, target_date: date) -> Optional[dict]:
    if not WEATHER_API_KEY:
        logger.error("WEATHER_API_KEY is not set")
        return None

    params = {
        "key": WEATHER_API_KEY,
        "q": f"{lat},{lon}",
        "dt": target_date.strftime("%Y-%m-%d"),
        "aqi": "no",
        "alerts": "no",
    }

    try:
        resp = requests.get(HISTORY_API_URL, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Weather API history request failed: %s", exc)
        return None

    payload = resp.json()
    location_data = payload.get("location", {})
    tz = location_data.get("tz_id")

    forecast_days = payload.get("forecast", {}).get("forecastday", [])
    if not forecast_days:
        logger.error("Weather history response missing forecast data")
        return None

    day_data = forecast_days[0]
    hours = day_data.get("hour", [])
    if not hours:
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

    return {"records": df.to_dict(orient="records"), "tz": tz}


# Backward compatibility helpers kept for existing notebooks/endpoints

def fetch_weather_forecast(latitude: float, longitude: float, target_date: date) -> Optional[List[dict]]:
    if not WEATHER_API_KEY:
        logger.error("WEATHER_API_KEY is not set")
        return None

    api_url = FORECAST_API_URL if target_date >= date.today() else HISTORY_API_URL

    logger.info(
        "Requesting weather data from %s for lat=%s lon=%s date=%s",
        api_url,
        latitude,
        longitude,
        target_date,
    )

    params = {
        "key": WEATHER_API_KEY,
        "q": f"{latitude},{longitude}",
        "dt": target_date.strftime("%Y-%m-%d"),
        "aqi": "no",
        "alerts": "no",
    }

    try:
        response = requests.get(api_url, params=params, timeout=15)
        logger.info("Weather API response status: %s", response.status_code)
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
