import math
from datetime import datetime
from typing import Iterable, Union

import numpy as np
import pandas as pd


SeriesLike = Union[pd.Series, Iterable[float], float]


def production_correction(temp_c: float, cloud_cover: float) -> float:
    """
    Изчислява коригиращ коефициент, отчитащ влиянието на температурата и облачността.
    """
    temp_coeff = -0.0044
    temp_diff = temp_c - 25  # отклонение от стандартната температура 25°C
    f_temp = 1 + temp_coeff * temp_diff + 0.0001 * (temp_diff ** 2)
    k = 1.0
    f_cloud = math.exp(-k * cloud_cover)
    return f_temp * f_cloud


def calculate_system_production(
    panel_power: float,
    temp_c: float,
    cloud_cover: float,
    num_panels: int,
    forecast_date: datetime,
    commissioning_date: datetime,
    degradation_rate: float = 0.0,
    string_loss_factor: float = 0.98,
    inverter_efficiency: float = 0.95,
) -> float:
    """
    Оставена за обратна съвместимост функция за изчисляване на производството на системата (W).
    """
    correction = production_correction(temp_c, cloud_cover)
    production_without_losses = panel_power * correction * num_panels
    production = production_without_losses * string_loss_factor * inverter_efficiency
    return production


def _to_series(values: SeriesLike, length: int, fill_value: float = 0.0) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    if isinstance(values, (list, tuple, np.ndarray)):
        return pd.Series(values)
    return pd.Series([values] * length)


def calculate_power_from_radiation(
    poa_w_m2: SeriesLike,
    temp_c: SeriesLike,
    panel_area_m2: float,
    module_efficiency: float,
    num_panels: int,
    forecast_times: Iterable[datetime],
    commissioning_date: datetime,
    degradation_rate: float = 0.0,
    string_loss_factor: float = 0.98,
    inverter_efficiency: float = 0.95,
    temp_coefficient: float = -0.004,
) -> pd.Series:
    """Convert plane-of-array irradiance to AC power (kW) for each timestamp."""
    times = pd.to_datetime(forecast_times)
    length = len(times)

    poa_series = _to_series(poa_w_m2, length).astype(float).clip(lower=0)
    temp_series = _to_series(temp_c, length).astype(float)

    base_dc_power = poa_series * panel_area_m2 * module_efficiency

    temperature_factor = 1 + temp_coefficient * (temp_series - 25)
    temperature_factor = temperature_factor.clip(lower=0)

    years_in_service = (times - pd.to_datetime(commissioning_date)).dt.days / 365.25
    yearly_degradation = max(float(degradation_rate), 0.0) / 100.0
    degradation_multiplier = (1 - yearly_degradation) ** years_in_service

    dc_power = base_dc_power * temperature_factor * num_panels * degradation_multiplier
    ac_power = dc_power * string_loss_factor * inverter_efficiency

    return ac_power.clip(lower=0) / 1000.0
