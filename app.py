from datetime import date, datetime, timedelta
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
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


def _get_default_tag() -> Optional[str]:
    available_tags = list_available_tags()
    if not available_tags:
        logger.warning("No available tags when attempting to select default")
        return None
    logger.info("Selecting default tag from %d available entries", len(available_tags))
    return available_tags[0].get("tag")


def _parse_health_target_date(target_date: Optional[str]) -> date:
    default_date = date.today() + timedelta(days=1)
    if target_date is None:
        return default_date

    try:
        return datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "Невалиден формат на дата. Очаква се YYYY-MM-DD.")


def _render_frontend_page() -> str:
    return """
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
            :root {
                color-scheme: light;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
            }

            body {
                background: #f4f6fb;
                margin: 0;
                padding: 0;
                color: #1f2933;
            }

            .page {
                max-width: 1000px;
                margin: 0 auto;
                padding: 32px 16px 48px;
            }

            h1 {
                font-size: 28px;
                margin-bottom: 16px;
            }

            p.helper {
                color: #5f6b7a;
                margin-top: 0;
                margin-bottom: 24px;
            }

            form {
                background: #fff;
                padding: 20px;
                border-radius: 16px;
                box-shadow: 0 10px 40px rgba(31, 41, 51, 0.08);
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 16px;
                align-items: end;
            }

            label {
                display: flex;
                flex-direction: column;
                gap: 8px;
                font-weight: 600;
                font-size: 14px;
            }

            select, input[type=date] {
                padding: 10px 12px;
                border-radius: 10px;
                border: 1px solid #d6d9e0;
                font-size: 14px;
            }

            button {
                padding: 12px 16px;
                border: none;
                border-radius: 10px;
                background: #2563eb;
                color: #fff;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.15s ease;
            }

            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }

            button:hover:not(:disabled) {
                background: #1d4ed8;
            }

            .card {
                background: #fff;
                padding: 20px;
                border-radius: 16px;
                box-shadow: 0 10px 40px rgba(31, 41, 51, 0.08);
                margin-top: 20px;
            }

            .chart-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 16px;
                align-items: stretch;
            }

            h3 {
                margin-top: 0;
                margin-bottom: 12px;
            }

            .status {
                margin-top: 12px;
                color: #5f6b7a;
                font-size: 14px;
            }
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

            <div class=\"chart-grid\">
                <div class=\"card\">
                    <h3>Прогноз за радиация (чисто небе)</h3>
                    <canvas id=\"radiation-chart\" height=\"120\"></canvas>
                </div>
                <div class=\"card\">
                    <h3>Прогнозна мощност при чисто небе</h3>
                    <canvas id=\"clearsky-power-chart\" height=\"120\"></canvas>
                </div>
                <div class=\"card\">
                    <h3>Температура и облачност</h3>
                    <canvas id=\"weather-chart\" height=\"120\"></canvas>
                </div>
                <div class=\"card\">
                    <h3>Прогнозна мощност с метео корекции</h3>
                    <canvas id=\"power-chart\" height=\"120\"></canvas>
                </div>
            </div>
            <div class=\"status\" id=\"status\">Заредете данните, за да видите графиките.</div>
            
            <div class=\"card\">
                <h3 style=\"margin-top: 0;\">Тестови проверки</h3>
                <p class=\"helper\" style=\"margin-bottom: 12px;\">Проверете връзката към таговете, метео API и заредения модел.</p>
                <div style=\"display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));\">
                    <button type=\"button\" id=\"test-tags\">Провери тагове</button>
                    <button type=\"button\" id=\"test-weather\">Провери Weather API</button>
                    <button type=\"button\" id=\"test-model\">Провери модел</button>
                </div>
                <div class=\"status\" id=\"test-status\" style=\"margin-top: 12px;\">Натиснете бутон, за да стартира тест.</div>
            </div>
        </div>

        <script>
            const tagSelect = document.getElementById('tag-select');
            const dateInput = document.getElementById('date-input');
            const form = document.getElementById('predict-form');
            const statusEl = document.getElementById('status');
            const submitBtn = document.getElementById('submit-btn');
            const testStatusEl = document.getElementById('test-status');
            const charts = {};

            function populateTags(tagOptions) {
                tagSelect.innerHTML = '';
                if (!tagOptions.length) {
                    const opt = document.createElement('option');
                    opt.textContent = 'Няма налични обекти';
                    opt.disabled = true;
                    opt.selected = true;
                    tagSelect.appendChild(opt);
                    submitBtn.disabled = true;
                    return;
                }

                tagOptions.forEach((item) => {
                    const opt = document.createElement('option');
                    const type = item.tag_type ? ' (' + item.tag_type + ')' : '';
                    opt.value = item.tag;
                    opt.textContent = item.tag + type;
                    tagSelect.appendChild(opt);
                });
            }

            async function loadTags() {
                tagSelect.innerHTML = '';
                const loadingOption = document.createElement('option');
                loadingOption.textContent = 'Зареждане на обекти...';
                loadingOption.disabled = true;
                loadingOption.selected = true;
                tagSelect.appendChild(loadingOption);
                submitBtn.disabled = true;

                try {
                    const response = await fetch('/tags');
                    if (!response.ok) {
                        throw new Error('Неуспешно зареждане на списъка с тагове');
                    }
                    const payload = await response.json();
                    const tagOptions = Array.isArray(payload.tags) ? payload.tags : [];
                    populateTags(tagOptions);
                    submitBtn.disabled = tagOptions.length === 0;
                } catch (err) {
                    console.error(err);
                    tagSelect.innerHTML = '';
                    const opt = document.createElement('option');
                    opt.textContent = err.message;
                    opt.disabled = true;
                    opt.selected = true;
                    tagSelect.appendChild(opt);
                    submitBtn.disabled = true;
                }
            }

            function setLoading(isLoading) {
                submitBtn.disabled = isLoading;
                submitBtn.textContent = isLoading ? 'Зареждане...' : 'Покажи прогноза';
            }

            function destroyChart(id) {
                if (charts[id]) {
                    charts[id].destroy();
                    delete charts[id];
                }
            }

            function renderLineChart({ id, labels, datasetLabel, data, color, yTitle, fillArea = true }) {
                const ctx = document.getElementById(id).getContext('2d');
                destroyChart(id);

                charts[id] = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels,
                        datasets: [
                            {
                                label: datasetLabel,
                                data,
                                fill: fillArea,
                                borderColor: color,
                                backgroundColor: fillArea ? `${color}1a` : 'transparent',
                                tension: 0.25,
                                pointRadius: 2,
                            },
                        ],
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: { legend: { display: true } },
                        scales: {
                            x: { ticks: { maxRotation: 45, minRotation: 45 } },
                            y: { beginAtZero: true, title: { display: true, text: yTitle } },
                        },
                    },
                });
            }

            function renderWeatherChart(labels, temps, clouds) {
                const ctx = document.getElementById('weather-chart').getContext('2d');
                destroyChart('weather-chart');

                charts['weather-chart'] = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels,
                        datasets: [
                            {
                                label: 'Температура (°C)',
                                data: temps,
                                borderColor: '#ef4444',
                                backgroundColor: '#ef44441a',
                                yAxisID: 'temp',
                                tension: 0.25,
                                pointRadius: 2,
                                fill: true,
                            },
                            {
                                label: 'Облачност (%)',
                                data: clouds,
                                borderColor: '#6b7280',
                                backgroundColor: '#6b72801a',
                                yAxisID: 'cloud',
                                tension: 0.25,
                                pointRadius: 2,
                                fill: true,
                            },
                        ],
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: { legend: { display: true } },
                        scales: {
                            x: { ticks: { maxRotation: 45, minRotation: 45 } },
                            temp: {
                                type: 'linear',
                                position: 'left',
                                title: { display: true, text: 'Температура (°C)' },
                            },
                            cloud: {
                                type: 'linear',
                                position: 'right',
                                title: { display: true, text: 'Облачност (%)' },
                                suggestedMin: 0,
                                suggestedMax: 100,
                                grid: { drawOnChartArea: false },
                            },
                        },
                    },
                });
            }

            function updateCharts(data) {
                const labels = data.map((p) => p.time);
                const clearskyRadiation = data.map((p) => p.radiation_poa_w_m2 ?? 0);
                const clearskyPower = data.map((p) => p.clearsky_power_kw ?? p.power_kw ?? 0);
                const adjustedPower = data.map((p) => p.power_kw ?? 0);
                const temps = data.map((p) => p.temp_c ?? 0);
                const clouds = data.map((p) => p.cloud ?? 0);

                renderLineChart({
                    id: 'radiation-chart',
                    labels,
                    datasetLabel: 'Радиация на панела (Вт/м²)',
                    data: clearskyRadiation,
                    color: '#f59e0b',
                    yTitle: 'Вт/м²',
                });

                renderLineChart({
                    id: 'clearsky-power-chart',
                    labels,
                    datasetLabel: 'Мощност при чисто небе (кВт)',
                    data: clearskyPower,
                    color: '#22c55e',
                    yTitle: 'кВт',
                });

                renderWeatherChart(labels, temps, clouds);

                renderLineChart({
                    id: 'power-chart',
                    labels,
                    datasetLabel: 'Прогнозна мощност с корекции (кВт)',
                    data: adjustedPower,
                    color: '#2563eb',
                    yTitle: 'кВт',
                });
            }

            form.addEventListener('submit', async (event) => {
                event.preventDefault();
                const tag = tagSelect.value;
                const prediction_date = dateInput.value;

                if (!tag || !prediction_date) {
                    statusEl.textContent = 'Посочете обект и дата.';
                    return;
                }

                setLoading(true);
                statusEl.textContent = 'Заявка за прогноза...';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ topic: tag, prediction_date }),
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Неуспешно получаване на прогноза');
                    }

                    const data = await response.json();
                    if (!Array.isArray(data) || data.length === 0) {
                        statusEl.textContent = 'Няма данни за избраните параметри.';
                        return;
                    }

                    statusEl.textContent = 'Получени са ' + data.length + ' точки. Графиките показват почасова прогноза.';
                    updateCharts(data);
                } catch (err) {
                    console.error(err);
                    statusEl.textContent = err.message;
                } finally {
                    setLoading(false);
                }
            });

            (function init() {
                loadTags();
                const today = new Date().toISOString().slice(0, 10);
                dateInput.value = today;
            })();

            async function runTest(path, label) {
                testStatusEl.textContent = `Изпълнява ${label}...`;
                try {
                    const response = await fetch(path);
                    if (!response.ok) {
                        const error = await response.json().catch(() => ({ detail: 'Неразпозната грешка' }));
                        throw new Error(error.detail || 'Неуспешна заявка');
                    }
                    const data = await response.json();
                    testStatusEl.textContent = `${label}: ${data.message || 'Успешно.'}`;
                } catch (err) {
                    console.error(err);
                    testStatusEl.textContent = `${label}: ${err.message}`;
                }
            }

            const testTomorrow = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
            const healthWeatherUrl = `/health/weather?lat=42.6977&lon=23.3219&target_date=${testTomorrow}`;

            document.getElementById('test-tags').addEventListener('click', () => runTest('/health/tags', 'Тест тагове'));
            document.getElementById('test-weather').addEventListener('click', () => runTest(healthWeatherUrl, 'Тест Weather API'));
            document.getElementById('test-model').addEventListener('click', () => runTest('/health/model', 'Тест модел'));
        </script>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    return _render_frontend_page()


@app.get("/tags")
def list_tags():
    tags = list_available_tags()
    logger.info("/tags requested, returning %d entries", len(tags))
    return {"tags": tags, "count": len(tags)}


@app.post("/predict")
def predict(request: PredictRequest):
    logger.info("/predict called for tag=%s date=%s", request.tag, request.prediction_date)
    try:
        forecast_date = datetime.strptime(request.prediction_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "Invalid date format. Expected YYYY-MM-DD.")

    tag = request.tag
    spec = get_tag_specification(tag)
    if not spec:
        logger.warning("No specification found for tag '%s'", tag)
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
        logger.error("Weather data retrieval failed for tag=%s date=%s", tag, forecast_date)
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
                "clearsky_power_kw": sanitize(float(ideal_power_kw.iloc[idx])),
                "temp_c": sanitize(float(row.get("temp_c", 0.0))),
                "cloud": sanitize(float(row.get("cloud", 0.0))),
                "radiation_poa_w_m2": sanitize(float(poa.iloc[idx])),
            }
        )

    return response


@app.get("/health/tags")
def health_tags():
    tags = list_available_tags()
    total = len(tags)
    sample = tags[0]["tag"] if tags else None
    logger.info("Health tags check: total=%d sample=%s", total, sample)
    return {
        "ok": bool(tags),
        "count": total,
        "sample_tag": sample,
        "message": f"Намерени тагове: {total}" if tags else "Няма налични тагове.",
    }


@app.get("/health/weather")
def health_weather(
    tag: Optional[str] = None,
    target_date: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
):
    forecast_date = _parse_health_target_date(target_date)

    if (lat is None) != (lon is None):
        raise HTTPException(400, "Необходимо е да подадете и latitude, и longitude.")

    chosen_tag = tag or _get_default_tag() if lat is None and lon is None else None

    if lat is None or lon is None:
        if not chosen_tag:
            raise HTTPException(503, "Не са намерени тагове за тестване.")

        spec = get_tag_specification(chosen_tag)
        if not spec:
            raise HTTPException(503, "Липсва спецификация за избрания таг.")

        lat = spec.get("latitude")
        lon = spec.get("longitude")
        if lat is None or lon is None:
            raise HTTPException(503, "Спецификацията няма координати.")

    data = fetch_weather_forecast(float(lat), float(lon), forecast_date)
    if not data:
        logger.error(
            "Health weather check failed for tag=%s date=%s lat=%s lon=%s",
            chosen_tag,
            forecast_date,
            lat,
            lon,
        )
        raise HTTPException(502, "Неуспешно извличане на прогноза.")

    return {
        "ok": True,
        "message": f"Получени записи: {len(data)}",
        "sample_time": data[0].get("time") if data else None,
        "tag_used": chosen_tag,
        "latitude": float(lat),
        "longitude": float(lon),
        "target_date": forecast_date.isoformat(),
    }


@app.get("/health/model")
def health_model():
    if MODEL is None:
        raise HTTPException(503, "Моделът не е зареден.")

    feature_names = getattr(MODEL, "feature_names_in_", None)
    message = "Моделът е зареден."
    if feature_names is not None:
        message += f" Брой признаци: {len(feature_names)}."

    return {
        "ok": True,
        "message": message,
    }
