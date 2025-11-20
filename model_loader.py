import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))


def _resolve_model_path(filename: str) -> str:
    if os.path.isabs(filename):
        return filename
    candidate = os.path.join(MODEL_DIR, filename)
    if os.path.exists(candidate):
        return candidate
    return filename


def load_model(model_name: str) -> Optional[object]:
    """Load a serialized model from disk.

    If ``model_name`` is provided without path, it is resolved relative to ``MODEL_DIR``.
    """
    if not model_name.endswith(".pkl"):
        model_name = f"{model_name}_model.pkl"

    model_path = _resolve_model_path(model_name)
    logger.debug("[load_model] Looking for model at %s", model_path)

    if not os.path.exists(model_path):
        logger.error("[load_model] Model file not found: %s", model_path)
        return None

    try:
        model = joblib.load(model_path)
        logger.info("[load_model] Loaded model from %s", model_path)
        return model
    except Exception as exc:
        logger.exception("[load_model] Failed to load model %s: %s", model_path, exc)
        return None


def predict_power(model, weather_sample: dict):
    """
    Backward-compatible prediction helper for legacy models.
    """
    try:
        n_features = getattr(model, "n_features_in_", None)
        logger.debug(f"[predict_power] model.n_features_in_ = {n_features}")

        if n_features == 2:
            rad_value = weather_sample.get("radiation_w_m2_y", 0.0)
            cloud_value = weather_sample.get("cloud", 0.0)

            input_data = pd.DataFrame({
                "radiation_w_m2_y": [rad_value],
                "cloud": [cloud_value]
            })
            logger.debug(f"[predict_power] DataFrame с 2 признака:\n{input_data}")
            prediction = model.predict(input_data)[0]
            logger.info(f"[predict_power] Прогноза с 2-признаков модел: {prediction}")
            return prediction

        elif n_features == 25:
            logger.debug(f"[predict_power] Получен weather_sample: {weather_sample}")
            features = [
                weather_sample.get("hour_local", 0.0),
                weather_sample.get("temp_c", 0.0),
                weather_sample.get("is_day", 0.0),
                weather_sample.get("condition_text", ""),
                weather_sample.get("wind_kph", 0.0),
                weather_sample.get("wind_degree", 0.0),
                weather_sample.get("wind_dir", ""),
                weather_sample.get("pressure_mb", 0.0),
                weather_sample.get("precip_mm", 0.0),
                weather_sample.get("snow_cm", 0.0),
                weather_sample.get("humidity", 0.0),
                weather_sample.get("cloud", 0.0),
                weather_sample.get("feelslike_c", 0.0),
                weather_sample.get("windchill_c", 0.0),
                weather_sample.get("heatindex_c", 0.0),
                weather_sample.get("dewpoint_c", 0.0),
                weather_sample.get("will_it_rain", 0.0),
                weather_sample.get("chance_of_rain", 0.0),
                weather_sample.get("will_it_snow", 0.0),
                weather_sample.get("chance_of_snow", 0.0),
                weather_sample.get("vis_km", 0.0),
                weather_sample.get("gust_kph", 0.0),
                weather_sample.get("uv", 0.0),
                weather_sample.get("hour_category", ""),
                weather_sample.get("solar_intensity_score", 0.0)
            ]

            columns = [
                "hour_local", "temp_c", "is_day", "condition_text", "wind_kph", "wind_degree",
                "wind_dir", "pressure_mb", "precip_mm", "snow_cm", "humidity", "cloud",
                "feelslike_c", "windchill_c", "heatindex_c", "dewpoint_c", "will_it_rain",
                "chance_of_rain", "will_it_snow", "chance_of_snow", "vis_km", "gust_kph", "uv",
                "hour_category", "solar_intensity_score"
            ]
            feature_df = pd.DataFrame([features], columns=columns)
            logger.debug(f"[predict_power] DataFrame с 25 признака:\n{feature_df}")
            prediction = model.predict(feature_df)[0]
            logger.info(f"[predict_power] Прогноза с 25-признаков модел: {prediction}")
            return prediction

        else:
            logger.error(f"[predict_power] Непознат брой признаци в модела: {n_features}")
            return None

    except Exception as e:
        logger.exception(f"[predict_power] Грешка при прогнозиране: {e}")
        return None
