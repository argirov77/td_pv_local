import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent / "Model"))


def load_model(tag: str) -> Optional[object]:
    """
    Зарежда модел за машинно обучение по даден tag.
    Ако моделът не е намерен, се опитва да зареди резервен модел "P0063H01_E001_model.pkl".
    """
    if tag.endswith(".pkl"):
        model_file = tag
    else:
        model_file = f"{tag}_model.pkl"

    model_path = os.path.join(MODEL_DIR, model_file)
    logger.debug("[load_model] Търсене на файл с модел: %s", model_path)

    if not os.path.exists(model_path):
        logger.error("[load_model] Файлът с модел не е намерен: %s", model_path)
        fallback_model = "P0063H01_E001_model.pkl"
        fallback_path = os.path.join(MODEL_DIR, fallback_model)
        if os.path.exists(fallback_path):
            try:
                logger.debug("[load_model] Опит за зареждане на резервен модел: %s", fallback_path)
                model = joblib.load(fallback_path)
                logger.info("[load_model] Резервен модел зареден успешно от: %s", fallback_path)
                return model
            except Exception as exc:
                logger.exception("[load_model] Грешка при зареждане на резервен модел: %s", exc)
                return None
        logger.error("[load_model] Резервният файл с модел не е намерен: %s", fallback_path)
        return None

    try:
        logger.debug("[load_model] Опит за зареждане на модел от: %s", model_path)
        model = joblib.load(model_path)
        logger.info("[load_model] Моделът е зареден успешно от: %s", model_path)
        return model
    except ModuleNotFoundError as mnfe:
        logger.error("[load_model] ModuleNotFoundError: %s. Проверете дали scikit-learn е инсталиран.", mnfe)
        return None
    except Exception as exc:
        logger.exception("[load_model] Грешка при зареждане на модел '%s': %s", tag, exc)
        return None


def predict_power(model, weather_sample: dict):
    """
    Генерира прогноза за мощност с модел, поддържащ 2 или 25 признака.
    """
    try:
        n_features = getattr(model, "n_features_in_", None)
        logger.debug("[predict_power] model.n_features_in_ = %s", n_features)

        if n_features == 2:
            rad_value = weather_sample.get("radiation_w_m2_y", 0.0)
            cloud_value = weather_sample.get("cloud", 0.0)

            input_data = pd.DataFrame({"radiation_w_m2_y": [rad_value], "cloud": [cloud_value]})
            logger.debug("[predict_power] DataFrame с 2 признака:\n%s", input_data)
            prediction = model.predict(input_data)[0]
            logger.info("[predict_power] Прогноза с 2-признаков модел: %s", prediction)
            return prediction

        if n_features == 25:
            logger.debug("[predict_power] Получен weather_sample: %s", weather_sample)
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
                weather_sample.get("solar_intensity_score", 0.0),
            ]

            columns = [
                "hour_local",
                "temp_c",
                "is_day",
                "condition_text",
                "wind_kph",
                "wind_degree",
                "wind_dir",
                "pressure_mb",
                "precip_mm",
                "snow_cm",
                "humidity",
                "cloud",
                "feelslike_c",
                "windchill_c",
                "heatindex_c",
                "dewpoint_c",
                "will_it_rain",
                "chance_of_rain",
                "will_it_snow",
                "chance_of_snow",
                "vis_km",
                "gust_kph",
                "uv",
                "hour_category",
                "solar_intensity_score",
            ]
            feature_df = pd.DataFrame([features], columns=columns)
            logger.debug("[predict_power] DataFrame с 25 признака:\n%s", feature_df)
            prediction = model.predict(feature_df)[0]
            logger.info("[predict_power] Прогноза с 25-признаков модел: %s", prediction)
            return prediction

        logger.error("[predict_power] Непознат брой признаци в модела: %s", n_features)
        return None

    except Exception as exc:
        logger.exception("[predict_power] Грешка при прогнозиране: %s", exc)
        return None
