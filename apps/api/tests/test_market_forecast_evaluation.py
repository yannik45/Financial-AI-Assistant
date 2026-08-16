import numpy as np
import pytest
from financial_ai.ml.market_forecast.evaluation.evaluation import (
    calculate_volatility_forecast_metrics,
)


def test_volatility_forecast_metrics_are_zero_for_perfect_predictions():
    actual = np.array([0.10, 0.20, 0.30])

    result = calculate_volatility_forecast_metrics(actual, actual.copy())

    assert result.mae == 0.0
    assert result.rmse == 0.0
    assert result.qlike == 0.0


def test_volatility_forecast_metrics_accept_array_like_inputs():
    result = calculate_volatility_forecast_metrics([0.10, 0.20], [0.10, 0.20])

    assert result.mae == 0.0


@pytest.mark.parametrize(
    ("actual", "predicted", "message"),
    [
        (np.array([]), np.array([]), "empty"),
        (np.array([0.10]), np.array([0.10, 0.20]), "same shape"),
        (np.array([0.10, np.nan]), np.array([0.10, 0.20]), "finite"),
        (np.array([0.10, 0.20]), np.array([0.10, np.inf]), "finite"),
        (np.array([0.00, 0.20]), np.array([0.10, 0.20]), "positive"),
        (np.array([0.10, 0.20]), np.array([-0.10, 0.20]), "positive"),
    ],
)
def test_volatility_forecast_metrics_reject_invalid_values(actual, predicted, message):
    with pytest.raises(ValueError, match=message):
        calculate_volatility_forecast_metrics(actual, predicted)
