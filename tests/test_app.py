from datetime import date, timedelta

import pytest
from fastapi import HTTPException

from app import MAX_FORECAST_DAYS_AHEAD, PredictRequest, predict


def test_predict_rejects_far_future_date():
    far_future_date = date.today() + timedelta(days=MAX_FORECAST_DAYS_AHEAD + 1)

    with pytest.raises(HTTPException) as excinfo:
        predict(
            PredictRequest(
                prediction_date=far_future_date.strftime("%Y-%m-%d"),
                topic="P0086H01/I002/Ptot",
            )
        )

    assert excinfo.value.status_code == 400
    assert "Прогнозата е достъпна" in str(excinfo.value.detail)
