from datetime import datetime
import logging
import math
from typing import Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_loader import load_model
from production import calculate_system_production
from radiation import calculate_panel_irradiance
from tag_spec_loader import get_tag_specification
from weather_api import fetch_weather_forecast

load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THRESHOLD_RADIATION = 40  # W/m²


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
    """Convert non-finite floats (NaN, inf) to None."""
    if isinstance(val, float) and not math.isfinite(val):
        return None
    return val


def _build_model_input(model, features: Dict) -> Optional[pd.DataFrame]:
    if model is None:
        return None

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        data = {name: [features.get(name)] for name in feature_names}
        return pd.DataFrame(data)

    n_features = getattr(model, "n_features_in_", None)
    if n_features == 2:
        return pd.DataFrame(
            {
                "radiation_w_m2_y": [features.get("radiation_w_m2_y", 0.0)],
                "cloud": [features.get("cloud", 0.0)],
            }
        )

    logger.warning("Unknown model input format; skipping ML adjustment")
    return None


@app.post("/predict")
def predict(request: PredictRequest):
    # 1) parse date
    try:
        forecast_date = datetime.strptime(request.prediction_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "Invalid date format. Expected YYYY-MM-DD.")

    # 2) fetch spec
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
    mlen = spec.get("module_length")
    mwid = spec.get("module_width")
    meff_pct = float(spec.get("module_efficiency") or spec.get("module_eff") or 17.7)
    panels_val = spec.get("total_panels")
    panels = int(panels_val) if panels_val is not None else None
    comm = spec.get("commissioning_date")
    degr = float(spec.get("degradation_rate", 0.0) or 0.0)

    if not mlen or not mwid:
        raise HTTPException(400, "Specification missing module dimensions.")

    panel_area = (float(mlen) / 1000) * (float(mwid) / 1000)
    mod_eff = float(meff_pct) / 100.0

    if panels is None:
        raise HTTPException(400, "Specification missing total_panels.")

    # 3) get weather records
    weather = fetch_weather_forecast(float(lat), float(lon), forecast_date)
    if not weather:
        raise HTTPException(404, "No weather data for this object/date.")

    # 4) load ML model
    model_name = tag.replace("/", "_") + "_model.pkl"
    model = load_model(model_name)

    # 5) iterate & compute
    result = []
    for rec in weather:
        tstr = rec.get("time")
        if not tstr:
            continue
        try:
            dt = datetime.strptime(tstr, "%Y-%m-%d %H:%M")
        except ValueError:
            continue

        irr = calculate_panel_irradiance(
            latitude=float(lat),
            longitude=float(lon),
            dt=dt,
            panel_tilt=tilt,
            panel_azimuth=azimuth,
            tz="Europe/Nicosia",
        )

        features = {
            "radiation_w_m2_y": irr,
            "cloud": float(rec.get("cloud", 0)),
            "hour_local": rec.get("hour_local"),
            "dayofyear": rec.get("dayofyear"),
            "hour_sin": rec.get("hour_sin"),
            "hour_cos": rec.get("hour_cos"),
            "doy_sin": rec.get("doy_sin"),
            "doy_cos": rec.get("doy_cos"),
            "temp_c": rec.get("temp_c"),
        }

        # apply threshold + optional ML correction
        if irr < THRESHOLD_RADIATION:
            eff = 0.0
        else:
            df_in = _build_model_input(model, features)
            if df_in is not None:
                eff = float(model.predict(df_in)[0])
            else:
                eff = irr

        base = eff * panel_area * mod_eff
        temp_c = float(rec.get("temp_c", 25))
        cloud_frac = float(rec.get("cloud", 0)) / 100.0

        try:
            commissioning_date = datetime.strptime(str(comm), "%Y-%m-%d")
        except Exception:
            commissioning_date = datetime.combine(forecast_date, datetime.min.time())

        power = calculate_system_production(
            panel_power=base,
            temp_c=temp_c,
            cloud_cover=cloud_frac,
            num_panels=panels,
            forecast_date=dt,
            commissioning_date=commissioning_date,
            degradation_rate=degr,
        )

        result.append({"x": dt.strftime("%Y-%m-%d %H:%M"), "y": sanitize(power)})

    return result
