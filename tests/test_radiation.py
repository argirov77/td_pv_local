import pandas as pd

from radiation import _ensure_datetime_index, calculate_clearsky_poa, calculate_panel_irradiance


def test_ensure_datetime_index_localizes_naive_times():
    times = ["2024-01-01 12:00"]
    tz = "Europe/Nicosia"

    result = _ensure_datetime_index(times, tz)

    assert isinstance(result, pd.DatetimeIndex)
    assert str(result.tz) == tz
    assert result[0].hour == 12


def test_calculate_clearsky_poa_daytime_positive_and_night_zero():
    time_index = pd.date_range("2024-06-21", periods=2, freq="12h", tz="UTC")

    poa = calculate_clearsky_poa(
        time_index=time_index,
        latitude=35.1856,
        longitude=33.3823,
        panel_tilt=30.0,
        panel_azimuth=180.0,
        tz="UTC",
    )

    assert len(poa) == 2
    assert poa.iloc[0] == 0.0
    assert poa.iloc[1] > 0.0
    assert poa.isna().sum() == 0


def test_calculate_panel_irradiance_returns_float():
    value = calculate_panel_irradiance(
        latitude=55.7558,
        longitude=37.6176,
        dt="2025-03-10 12:00:00",
        panel_tilt=30.0,
        panel_azimuth=180.0,
        tz="Europe/Moscow",
    )

    assert isinstance(value, float)
    assert value > 0
