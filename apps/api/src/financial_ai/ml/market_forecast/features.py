"""Backward-looking features for daily market volatility forecasts."""

import numpy as np
import pandas as pd

from financial_ai.ml.market_forecast.daily_bars import validate_daily_bars

FEATURE_COLUMNS = (
    "log_return_1d",
    "absolute_log_return_1d",
    "realized_volatility_5d",
    "realized_volatility_20d",
    "realized_volatility_60d",
    "momentum_5d",
    "momentum_20d",
    "intraday_range_1d",
    "intraday_range_mean_20d",
    "volume_ratio_5d_20d",
)


def build_market_features(daily_bars: pd.DataFrame) -> pd.DataFrame:
    """Return daily bars with scale-independent, backward-looking features."""
    validated = validate_daily_bars(daily_bars)
    prices_by_symbol = validated.groupby("symbol", sort=False)["adjusted_close"]
    validated["log_return_1d"] = prices_by_symbol.transform(
        lambda prices: np.log(prices / prices.shift(1))
    )
    validated["absolute_log_return_1d"] = validated["log_return_1d"].abs()
    validated["momentum_5d"] = prices_by_symbol.transform(
        lambda prices: np.log(prices / prices.shift(5))
    )
    validated["momentum_20d"] = prices_by_symbol.transform(
        lambda prices: np.log(prices / prices.shift(20))
    )

    returns_by_symbol = validated.groupby("symbol", sort=False)["log_return_1d"]

    for window in 5, 20, 60:
        validated[f"realized_volatility_{window}d"] = returns_by_symbol.transform(
            lambda returns, current_window=window: returns.rolling(
                window=current_window,
                min_periods=current_window,
            ).std(ddof=1)
        ) * np.sqrt(252)

    validated["intraday_range_1d"] = np.log(validated["high"] / validated["low"])

    range_by_symbol = validated.groupby("symbol", sort=False)["intraday_range_1d"]
    validated["intraday_range_mean_20d"] = range_by_symbol.transform(
        lambda ranges: ranges.rolling(window=20, min_periods=20).mean()
    )

    volume_by_symbol = validated.groupby("symbol", sort=False)["volume"]
    average_volume_5d = volume_by_symbol.transform(
        lambda volume: volume.rolling(window=5, min_periods=5).mean()
    )
    average_volume_20d = volume_by_symbol.transform(
        lambda volume: volume.rolling(window=20, min_periods=20).mean()
    )
    validated["volume_ratio_5d_20d"] = average_volume_5d / average_volume_20d.replace(
        0, np.nan
    )

    return validated.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)
