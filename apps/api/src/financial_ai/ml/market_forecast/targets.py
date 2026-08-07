"""Forward-looking target construction for market volatility forecasts."""

import numpy as np
import pandas as pd

from financial_ai.ml.market_forecast.daily_bars import validate_daily_bars

DEFAULT_VOLATILITY_HORIZON = 20
TRADING_DAYS_PER_YEAR = 252
TARGET_COLUMN = "forward_realized_volatility_20d"


def build_forward_volatility_target(
    daily_bars: pd.DataFrame,
    horizon: int = DEFAULT_VOLATILITY_HORIZON,
) -> pd.DataFrame:
    """Return daily bars with annualized forward realized volatility targets."""
    if horizon < 2:
        raise ValueError("Volatility horizon must contain at least two trading days")

    validated = validate_daily_bars(daily_bars)
    daily_log_returns = validated.groupby("symbol", sort=False)["adjusted_close"].transform(
        lambda prices: np.log(prices / prices.shift(1))
    )

    def forward_standard_deviation(returns: pd.Series) -> pd.Series:
        future_returns = returns.shift(-1)
        return (
            future_returns.iloc[::-1]
            .rolling(window=horizon, min_periods=horizon)
            .std(ddof=1)
            .iloc[::-1]
        )

    forward_volatility = daily_log_returns.groupby(validated["symbol"], sort=False).transform(
        forward_standard_deviation
    )
    validated[TARGET_COLUMN] = forward_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    return validated.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
