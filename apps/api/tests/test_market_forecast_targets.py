import numpy as np
import pandas as pd
import pytest
from financial_ai.ml.market_forecast.targets import (
    TARGET_COLUMN,
    build_forward_volatility_target,
)


def daily_bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "observed_on": pd.date_range("2024-01-02", periods=len(closes), freq="B"),
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "adjusted_close": closes,
            "volume": [1_000_000] * len(closes),
        }
    )


def test_forward_volatility_uses_only_the_next_horizon_returns():
    closes = [100, 102, 101, 105, 103, 107]
    result = build_forward_volatility_target(daily_bars("AAPL", closes), horizon=3)

    expected_returns = np.diff(np.log(closes[:4]))
    expected = expected_returns.std(ddof=1) * np.sqrt(252)

    assert len(result) == 3
    assert result.iloc[0][TARGET_COLUMN] == pytest.approx(expected)
    assert result.iloc[0]["observed_on"] == pd.Timestamp("2024-01-02")


def test_forward_volatility_is_calculated_independently_per_symbol():
    stable = daily_bars("STABLE", [100] * 6)
    variable = daily_bars("VARIABLE", [100, 110, 90, 115, 85, 120])

    result = build_forward_volatility_target(
        pd.concat([variable, stable], ignore_index=True),
        horizon=3,
    )

    stable_targets = result.loc[result["symbol"] == "STABLE", TARGET_COLUMN]
    variable_targets = result.loc[result["symbol"] == "VARIABLE", TARGET_COLUMN]
    assert stable_targets.eq(0).all()
    assert variable_targets.gt(0).all()


@pytest.mark.parametrize("horizon", [0, 1, -5])
def test_forward_volatility_rejects_invalid_horizons(horizon):
    with pytest.raises(ValueError, match="horizon"):
        build_forward_volatility_target(daily_bars("AAPL", [100, 101]), horizon=horizon)
