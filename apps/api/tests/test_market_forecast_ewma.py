import numpy as np
import pandas as pd
import pytest
from financial_ai.ml.market_forecast.ewma import (
    EWMA_VOLATILITY_COLUMN,
    build_ewma_volatility,
)


def daily_bars() -> pd.DataFrame:
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    closes = 100 * np.exp(np.r_[0.0, returns].cumsum())
    dates = pd.date_range("2024-01-02", periods=len(closes), freq="B")
    frames = []
    for symbol, scale in (("AAPL", 1.0), ("MSFT", 2.0)):
        prices = closes * scale
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "observed_on": dates,
                    "open": prices,
                    "high": prices * 1.01,
                    "low": prices * 0.99,
                    "close": prices,
                    "adjusted_close": prices,
                    "volume": 1_000_000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_ewma_volatility_is_grouped_annualized_and_backward_looking():
    result = build_ewma_volatility(daily_bars(), decay=0.5, min_observations=2)

    aapl = result.loc[result["symbol"] == "AAPL"].reset_index(drop=True)
    expected_last_variance = (
        0.5 * (0.5 * (0.5 * 0.02**2 + 0.5 * 0.01**2) + 0.5 * 0.03**2) + 0.5 * 0.02**2
    )
    expected_last_variance = 0.5 * expected_last_variance + 0.5 * 0.01**2
    assert aapl[EWMA_VOLATILITY_COLUMN].iloc[-1] == pytest.approx(
        np.sqrt(expected_last_variance * 252)
    )
    assert result.groupby("symbol").size().to_dict() == {"AAPL": 4, "MSFT": 4}


def test_ewma_volatility_does_not_modify_source_bars():
    source = daily_bars()
    expected = source.copy(deep=True)

    build_ewma_volatility(source, min_observations=2)

    pd.testing.assert_frame_equal(source, expected)


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, 1.1])
def test_ewma_volatility_rejects_invalid_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        build_ewma_volatility(daily_bars(), decay=decay)
