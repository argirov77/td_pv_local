from datetime import date, datetime
import logging
import math
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd

from tag_spec_loader import get_tag_specification, list_available_tags
from radiation import calculate_panel_irradiance
from production import calculate_system_production
from model_loader import load_model
from weather_api import extract_weather_history, get_forecast_by_coords

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THRESHOLD_RADIATION = 40  # W/m²


class PredictRequest(BaseModel):
    prediction_date: str = Field(..., description="Дата във формат YYYY-MM-DD")
    tag: str = Field(..., alias="topic", description="Идентификатор на таг (или топик)")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "prediction_date": "2025-03-20",
                "topic": "P0086H01/I002/Ptot",
            }
        }


def sanitize(val):
    if isinstance(val, float) and not math.isfinite(val):
        return None
    return val


FRONTEND_PATH = Path(__file__).with_name("frontend.html")
FRONTEND_HTML = FRONTEND_PATH.read_text(encoding="utf-8") if FRONTEND_PATH.exists() else "<h1>Frontend not found</h1>"


@app.get("/", response_class=HTMLResponse)
def home():
    return FRONTEND_HTML


@app.get("/tags")
def list_tags():
    tags = list_available_tags()
    return {"tags": tags, "count": len(tags)}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        forecast_date = datetime.strptime(request.prediction_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "Невалиден формат на дата. Очаква се YYYY-MM-DD.")

    today = date.today()
    tag = request.tag
    spec = get_tag_specification(tag)
    if not spec:
        raise HTTPException(400, f"Няма спецификация за таг '{tag}'.")

    uid = spec.get("sm_user_object_id")
    if not uid:
        raise HTTPException(400, "Липсва 'sm_user_object_id' в спецификацията.")

    lat = spec.get("latitude")
    lon = spec.get("longitude")
    if lat is None or lon is None:
        raise HTTPException(400, "Липсват координати (latitude/longitude) в спецификацията.")

    tilt = float(spec.get("tilt", 0.0) or 0.0)
    azimuth = float(spec.get("azimuth", 180.0) or 180.0)
    mlen = spec.get("module_length") or spec.get("module_height")
    mwid = spec.get("module_width")
    meff_pct = float(spec.get("module_efficiency", 17.7) or 17.7)
    panels = spec.get("total_panels")
    comm = spec.get("commissioning_date")
    degr = float(spec.get("degradation_rate", 0.0) or 0.0)

    if not mlen or not mwid:
        raise HTTPException(400, "Липсват размери на панела в спецификацията.")
    if panels is None:
        raise HTTPException(400, "Липсва стойност за total_panels в спецификацията.")

    panel_area = (float(mlen) / 1000) * (float(mwid) / 1000)
    mod_eff = meff_pct / 100.0
    commissioning_date = (
        datetime.strptime(str(comm), "%Y-%m-%d")
        if comm
        else datetime.combine(forecast_date, datetime.min.time())
    )
    series = {
        "time": [],
        "radiation": [],
        "cloud": [],
        "temperature": [],
        "ideal_power": [],
        "predicted_power": [],
    }

    if forecast_date >= today:
        weather = get_forecast_by_coords(float(lat), float(lon), forecast_date)
        if not weather:
            raise HTTPException(502, "Грешка при вземане на прогноза от WeatherAPI")

        for t, temp_c, cloud in zip(weather["time"], weather["temp_c"], weather["cloud"]):
            dt = datetime.strptime(t, "%Y-%m-%d %H:%M")
            irr = calculate_panel_irradiance(
                latitude=float(lat),
                longitude=float(lon),
                dt=dt,
                panel_tilt=tilt,
                panel_azimuth=azimuth,
                tz="Europe/Nicosia",
            )

            eff = 0.0 if irr < THRESHOLD_RADIATION else irr
            base = eff * panel_area * mod_eff
            temp_c_val = float(temp_c) if temp_c is not None else 25.0
            cloud_percent = float(cloud) if cloud is not None else 0.0
            cloud_frac = cloud_percent / 100.0

            predicted_power = calculate_system_production(
                panel_power=base,
                temp_c=temp_c_val,
                cloud_cover=cloud_frac,
                num_panels=int(panels),
                forecast_date=dt,
                commissioning_date=commissioning_date,
                degradation_rate=degr,
            )

            ideal_power = calculate_system_production(
                panel_power=base,
                temp_c=25.0,
                cloud_cover=0.0,
                num_panels=int(panels),
                forecast_date=dt,
                commissioning_date=commissioning_date,
                degradation_rate=degr,
            )

            series["time"].append(dt.strftime("%Y-%m-%d %H:%M"))
            series["radiation"].append(sanitize(irr))
            series["cloud"].append(sanitize(cloud_percent))
            series["temperature"].append(sanitize(temp_c_val))
            series["ideal_power"].append(sanitize(ideal_power))
            series["predicted_power"].append(sanitize(predicted_power))

        return series

    weather_history = extract_weather_history(float(lat), float(lon), forecast_date)
    if not weather_history:
        raise HTTPException(404, "Няма метео-данни за този обект/дата.")

    model_name = tag.replace("/", "_") + "_model.pkl"
    model = load_model(model_name)

    for rec in weather_history:
        tstr = rec.get("time")
        try:
            dt = datetime.strptime(tstr, "%Y-%m-%d %H:%M") if tstr else datetime.combine(forecast_date, datetime.min.time())
        except Exception:
            dt = datetime.combine(forecast_date, datetime.min.time())

        irr = calculate_panel_irradiance(
            latitude=float(lat),
            longitude=float(lon),
            dt=dt,
            panel_tilt=tilt,
            panel_azimuth=azimuth,
            tz="Europe/Nicosia",
        )

        if irr < THRESHOLD_RADIATION:
            eff = 0.0
        else:
            if model:
                df_in = pd.DataFrame(
                    {
                        "radiation_w_m2_y": [irr],
                        "cloud": [float(rec.get("cloud", 0))],
                    }
                )
                eff = float(model.predict(df_in)[0])
            else:
                eff = irr

        base = eff * panel_area * mod_eff
        temp_c_val = float(rec.get("temp_c", 25))
        cloud_percent = float(rec.get("cloud", 0))
        cloud_frac = cloud_percent / 100.0

        predicted_power = calculate_system_production(
            panel_power=base,
            temp_c=temp_c_val,
            cloud_cover=cloud_frac,
            num_panels=int(panels),
            forecast_date=dt,
            commissioning_date=commissioning_date,
            degradation_rate=degr,
        )

        ideal_power = calculate_system_production(
            panel_power=base,
            temp_c=25.0,
            cloud_cover=0.0,
            num_panels=int(panels),
            forecast_date=dt,
            commissioning_date=commissioning_date,
            degradation_rate=degr,
        )

        series["time"].append(dt.strftime("%Y-%m-%d %H:%M"))
        series["radiation"].append(sanitize(irr))
        series["cloud"].append(sanitize(cloud_percent))
        series["temperature"].append(sanitize(temp_c_val))
        series["ideal_power"].append(sanitize(ideal_power))
        series["predicted_power"].append(sanitize(predicted_power))

    return series


@app.get("/weather/forecast")
def weather_forecast(
    lat: float = Query(..., description="Ширина"),
    lon: float = Query(..., description="Дължина"),
    date_str: str = Query(..., description="Дата във формат YYYY-MM-DD"),
):
    try:
        forecast_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "Невалиден формат на дата. Използвай YYYY-MM-DD.")

    data = get_forecast_by_coords(lat, lon, forecast_date)
    if not data:
        raise HTTPException(502, "Грешка при вземане на данни от WeatherAPI")

    return data
