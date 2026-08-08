"""Exponentially weighted volatility reference forecasts."""

import numpy as np
import pandas as pd

from financial_ai.ml.market_forecast.daily_bars import validate_daily_bars

DEFAULT_EWMA_DECAY = 0.94
DEFAULT_EWMA_MIN_OBSERVATIONS = 20
EWMA_VOLATILITY_COLUMN = "ewma_volatility"


def build_ewma_volatility(
    daily_bars: pd.DataFrame,
    *,
    decay: float = DEFAULT_EWMA_DECAY,
    min_observations: int = DEFAULT_EWMA_MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Calculate backward-looking annualized EWMA volatility by symbol."""
    if not 0 < decay < 1:
        raise ValueError("EWMA decay must be between zero and one")
    if min_observations < 2:
        raise ValueError("EWMA requires at least two observations")

    result = validate_daily_bars(daily_bars)
    prices_by_symbol = result.groupby("symbol", sort=False)["adjusted_close"]
    result["log_return_1d"] = prices_by_symbol.transform(
        lambda prices: np.log(prices / prices.shift(1))
    )
    returns_by_symbol = result.groupby("symbol", sort=False)["log_return_1d"]
    ewma_variance = returns_by_symbol.transform(
        lambda returns: (
            returns.pow(2)
            .ewm(
                alpha=1 - decay,
                adjust=False,
                min_periods=min_observations,
            )
            .mean()
        )
    )
    result[EWMA_VOLATILITY_COLUMN] = np.sqrt(ewma_variance * 252)
    return (
        result.dropna(subset=[EWMA_VOLATILITY_COLUMN])
        .loc[:, ["symbol", "observed_on", EWMA_VOLATILITY_COLUMN]]
        .reset_index(drop=True)
    )
