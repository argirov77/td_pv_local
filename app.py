from datetime import datetime
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from model_loader import load_model
from production import calculate_power_from_radiation
from radiation import calculate_clearsky_poa
from tag_spec_loader import get_tag_specification, list_available_tags
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


def _render_frontend_page() -> str:
    tags = list_available_tags()
    tags_json = json.dumps(tags)

    return f"""
    <!DOCTYPE html>
    <html lang=\"bg\">
    <head>
        <meta charset=\"UTF-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
        <title>Прогноза за производство</title>
        <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />
        <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />
        <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap\" rel=\"stylesheet\" />
        <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
        <style>
            :root {{
                color-scheme: light;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
            }}

            body {{
                background: #f4f6fb;
                margin: 0;
                padding: 0;
                color: #1f2933;
            }}

            .page {{
                max-width: 1000px;
                margin: 0 auto;
                padding: 32px 16px 48px;
            }}

            h1 {{
                font-size: 28px;
                margin-bottom: 16px;
            }}

            p.helper {{
                color: #5f6b7a;
                margin-top: 0;
                margin-bottom: 24px;
            }}

            form {{
                background: #fff;
                padding: 20px;
                border-radius: 16px;
                box-shadow: 0 10px 40px rgba(31, 41, 51, 0.08);
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 16px;
                align-items: end;
            }}

            label {{
                display: flex;
                flex-direction: column;
                gap: 8px;
                font-weight: 600;
                font-size: 14px;
            }}

            select, input[type=date] {{
                padding: 10px 12px;
                border-radius: 10px;
                border: 1px solid #d6d9e0;
                font-size: 14px;
            }}

            button {{
                padding: 12px 16px;
                border: none;
                border-radius: 10px;
                background: #2563eb;
                color: #fff;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.15s ease;
            }}

            button:disabled {{
                opacity: 0.6;
                cursor: not-allowed;
            }}

            button:hover:not(:disabled) {{
                background: #1d4ed8;
            }}

            .card {{
                background: #fff;
                padding: 20px;
                border-radius: 16px;
                box-shadow: 0 10px 40px rgba(31, 41, 51, 0.08);
                margin-top: 20px;
            }}

            .status {{
                margin-top: 12px;
                color: #5f6b7a;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class=\"page\">
            <h1>Прогноза за производство</h1>
            <p class=\"helper\">Изберете обект и дата, за да получите почасова графика на прогнозираната мощност.</p>
            <form id=\"predict-form\">
                <label>
                    Обект
                    <select id=\"tag-select\" required></select>
                </label>
                <label>
                    Дата
                    <input id=\"date-input\" type=\"date\" required />
                </label>
                <button type=\"submit\" id=\"submit-btn\">Покажи прогноза</button>
            </form>

            <div class=\"card\">
                <canvas id=\"chart\" height=\"120\"></canvas>
                <div class=\"status\" id=\"status\">Заредете данните, за да видите графиката.</div>
            </div>
        </div>

        <script>
            const tagOptions = {tags_json};
            const tagSelect = document.getElementById('tag-select');
            const dateInput = document.getElementById('date-input');
            const form = document.getElementById('predict-form');
            const statusEl = document.getElementById('status');
            const submitBtn = document.getElementById('submit-btn');
            let chartInstance = null;

            function populateTags() {{
                tagSelect.innerHTML = '';
                if (!tagOptions.length) {{
                    const opt = document.createElement('option');
                    opt.textContent = 'Няма налични обекти';
                    opt.disabled = true;
                    opt.selected = true;
                    tagSelect.appendChild(opt);
                    submitBtn.disabled = true;
                    return;
                }}

                tagOptions.forEach((item) => {{
                    const opt = document.createElement('option');
                    const type = item.tag_type ? ' (' + item.tag_type + ')' : '';
                    opt.value = item.tag;
                    opt.textContent = item.tag + type;
                    tagSelect.appendChild(opt);
                }});
            }}

            function setLoading(isLoading) {{
                submitBtn.disabled = isLoading;
                submitBtn.textContent = isLoading ? 'Зареждане...' : 'Покажи прогноза';
            }}

            function updateChart(data) {{
                const labels = data.map((p) => p.time);
                const values = data.map((p) => p.power_kw);

                const ctx = document.getElementById('chart').getContext('2d');
                if (chartInstance) {{
                    chartInstance.destroy();
                }}

                chartInstance = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels,
                        datasets: [{{
                            label: 'Прогноза за мощност (кВт)',
                            data: values,
                            fill: true,
                            borderColor: '#2563eb',
                            backgroundColor: 'rgba(37, 99, 235, 0.1)',
                            tension: 0.25,
                            pointRadius: 2,
                        }}],
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {{
                            legend: {{ display: true }},
                        }},
                        scales: {{
                            x: {{
                                ticks: {{ maxRotation: 45, minRotation: 45 }},
                            }},
                            y: {{ beginAtZero: true, title: {{ display: true, text: 'кВт' }} }},
                        }},
                    }},
                }});
            }}

            form.addEventListener('submit', async (event) => {{
                event.preventDefault();
                const tag = tagSelect.value;
                const prediction_date = dateInput.value;

                if (!tag || !prediction_date) {{
                    statusEl.textContent = 'Посочете обект и дата.';
                    return;
                }}

                setLoading(true);
                statusEl.textContent = 'Заявка за прогноза...';

                try {{
                    const response = await fetch('/predict', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ topic: tag, prediction_date }}),
                    }});

                    if (!response.ok) {{
                        const error = await response.json();
                        throw new Error(error.detail || 'Неуспешно получаване на прогноза');
                    }}

                    const data = await response.json();
                    if (!Array.isArray(data) || data.length === 0) {{
                        statusEl.textContent = 'Няма данни за избраните параметри.';
                        return;
                    }}

                    statusEl.textContent = 'Получени са ' + data.length + ' точки. Графиката показва почасова прогноза.';
                    updateChart(data);
                }} catch (err) {{
                    console.error(err);
                    statusEl.textContent = err.message;
                }} finally {{
                    setLoading(false);
                }}
            }});

            (function init() {{
                populateTags();
                const today = new Date().toISOString().slice(0, 10);
                dateInput.value = today;
            }})();
        </script>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    return _render_frontend_page()


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
