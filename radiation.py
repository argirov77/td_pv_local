import numpy as np
import pandas as pd
import pvlib
import math
from pvlib.location import Location
from typing import Iterable, Union


TimeLike = Union[pd.DatetimeIndex, Iterable[pd.Timestamp], Iterable[str]]


def _ensure_datetime_index(times: TimeLike, tz: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(times)
    if index.tz is None:
        index = index.tz_localize(tz)
    else:
        index = index.tz_convert(tz)
    return index


def calculate_clearsky_poa(
    time_index: TimeLike,
    latitude: float,
    longitude: float,
    panel_tilt: float,
    panel_azimuth: float,
    tz: str = "UTC",
) -> pd.Series:
    """Calculate clearsky plane-of-array irradiance for a series of timestamps.

    Parameters
    ----------
    time_index: iterable of timestamps or a ``DatetimeIndex``
        96-point time series for the target day (15-minute resolution).
    latitude, longitude: float
        Site coordinates.
    panel_tilt, panel_azimuth: float
        Panel geometry in degrees.
    tz: str
        Timezone to localize naive timestamps.

    Returns
    -------
    pd.Series
        POA irradiance in W/m² for each timestamp (nighttime values set to 0).
    """
    if panel_azimuth is None or not math.isfinite(panel_azimuth):
        panel_azimuth = 180.0

    if not math.isfinite(panel_tilt):
        panel_tilt = 0.0

    times = _ensure_datetime_index(time_index, tz)
    site = Location(latitude, longitude, tz=times.tz)

    solpos = site.get_solarposition(times)
    clearsky = site.get_clearsky(times)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=panel_tilt,
        surface_azimuth=panel_azimuth,
        solar_zenith=solpos["zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )["poa_global"]

    poa = poa.where(solpos["apparent_zenith"] < 90, 0.0)
    poa = poa.fillna(0.0)
    return poa.astype(float)


def calculate_panel_irradiance(
    latitude: float,
    longitude: float,
    dt,
    panel_tilt: float,
    panel_azimuth: float,
    tz: str = "Europe/Nicosia",
) -> float:
    """Backward-compatible wrapper for single-timestamp POA computation."""
    try:
        poa_series = calculate_clearsky_poa([pd.Timestamp(dt)], latitude, longitude, panel_tilt, panel_azimuth, tz)
        value = float(poa_series.iloc[0])
        if value is None or np.isnan(value):
            return 0.0
        return value
    except Exception:
        return 0.0
