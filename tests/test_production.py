from datetime import datetime

import numpy as np
import pandas as pd

from production import _to_series, calculate_power_from_radiation


def test_to_series_expands_scalar():
    result = _to_series(5.0, length=3)

    assert isinstance(result, pd.Series)
    assert result.tolist() == [5.0, 5.0, 5.0]


def test_calculate_power_from_radiation_applies_temperature_and_degradation():
    poa = [800.0, 0.0]
    temperatures = [25.0, 35.0]
    times = ["2024-01-01 12:00", "2024-01-01 13:00"]

    power = calculate_power_from_radiation(
        poa_w_m2=poa,
        temp_c=temperatures,
        panel_area_m2=2.0,
        module_efficiency=0.20,
        num_panels=10,
        forecast_times=times,
        commissioning_date=datetime(2023, 1, 1),
        degradation_rate=1.0,
        string_loss_factor=0.98,
        inverter_efficiency=0.95,
    )

    assert isinstance(power, pd.Series)
    assert len(power) == 2
    assert power.iloc[1] == 0.0

    expected_first = 2.95  # derived from manual calculation (~2.949 kW)
    np.testing.assert_allclose(power.iloc[0], expected_first, rtol=1e-3)
