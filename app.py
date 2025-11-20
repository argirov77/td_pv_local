from datetime import datetime
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_loader import load_model
from production import calculate_power_from_radiation
from radiation import calculate_clearsky_poa
from tag_spec_loader import get_tag_specification
from weather_api import fetch_weather_forecast

load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = load_model("rf_model_v3.pkl")


class PredictRequest(BaseModel):
    prediction_date: str = Field(..., description="Date in YYYY-MM-DD format")
    tag: str = Field(..., alias="topic", description="Tag (or topic) identifier")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "prediction_date": "2025-03-20",
                "topic": "P0086H01/I002/Ptot",
            }
        }


def sanitize(val):
    if isinstance(val, float) and not np.isfinite(val):
        return None
    return val


def _parse_commissioning_date(raw_value, fallback_date: datetime) -> datetime:
    try:
        return datetime.strptime(str(raw_value), "%Y-%m-%d")
    except Exception:
        return fallback_date


def _build_feature_frame(weather: pd.DataFrame, poa: pd.Series, model) -> Optional[pd.DataFrame]:
    if weather.empty or model is None:
        return None

    features = weather.copy()
    features["poa_w_m2"] = poa.values

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return None

    for name in feature_names:
        if name not in features.columns:
            features[name] = 0.0

    return features[list(feature_names)]


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        forecast_date = datetime.strptime(request.prediction_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "Invalid date format. Expected YYYY-MM-DD.")

    tag = request.tag
    spec = get_tag_specification(tag)
    if not spec:
        raise HTTPException(400, f"No specification found for tag '{tag}'.")

    lat = spec.get("latitude")
    lon = spec.get("longitude")
    if lat is None or lon is None:
        raise HTTPException(400, "Specification missing latitude/longitude.")

    tilt = float(spec.get("tilt", 0.0) or 0.0)
    azimuth = float(spec.get("azimuth", 180.0) or 180.0)
    mlen = spec.get("module_length") or spec.get("module_height")
    mwid = spec.get("module_width")
    mod_eff_pct = float(spec.get("module_efficiency") or spec.get("module_eff") or 17.7)
    panels_val = spec.get("total_panels")
    panels = int(panels_val) if panels_val is not None else None
    comm = spec.get("commissioning_date")
    degr = float(spec.get("degradation_rate", 0.0) or 0.0)
    timezone_name = spec.get("timezone") or "UTC"

    if not mlen or not mwid:
        raise HTTPException(400, "Specification missing module dimensions.")
    if panels is None:
        raise HTTPException(400, "Specification missing total_panels.")

    panel_area = (float(mlen) / 1000) * (float(mwid) / 1000)
    mod_eff = float(mod_eff_pct) / 100.0

    weather_records = fetch_weather_forecast(float(lat), float(lon), forecast_date)
    if not weather_records:
        raise HTTPException(404, "No weather data for this object/date.")

    weather_df = pd.DataFrame(weather_records)
    if weather_df.empty:
        raise HTTPException(404, "No weather data for this object/date.")

    weather_df["timestamp"] = pd.to_datetime(weather_df["time"])

    poa = calculate_clearsky_poa(
        weather_df["timestamp"],
        latitude=float(lat),
        longitude=float(lon),
        panel_tilt=tilt,
        panel_azimuth=azimuth,
        tz=timezone_name,
    )

    commissioning_date = _parse_commissioning_date(comm, datetime.combine(forecast_date, datetime.min.time()))

    ideal_power_kw = calculate_power_from_radiation(
        poa_w_m2=poa,
        temp_c=weather_df.get("temp_c", 25),
        panel_area_m2=panel_area,
        module_efficiency=mod_eff,
        num_panels=panels,
        forecast_times=weather_df["timestamp"],
        commissioning_date=commissioning_date,
        degradation_rate=degr,
    )

    model_input = _build_feature_frame(weather_df, poa, MODEL)
    if model_input is not None and MODEL is not None:
        predictions = pd.Series(MODEL.predict(model_input))
        if predictions.between(0, 1.1).all():
            power_kw = ideal_power_kw * predictions.clip(lower=0)
        else:
            power_kw = predictions
    else:
        power_kw = ideal_power_kw

    response = []
    for idx, row in weather_df.iterrows():
        response.append(
            {
                "time": row["time"],
                "power_kw": sanitize(float(power_kw.iloc[idx])),
                "temp_c": sanitize(float(row.get("temp_c", 0.0))),
                "cloud": sanitize(float(row.get("cloud", 0.0))),
                "radiation_poa_w_m2": sanitize(float(poa.iloc[idx])),
            }
        )

    return response
