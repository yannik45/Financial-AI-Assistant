"""Inference input preparation for market volatility forecasts."""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from financial_ai.ml.market_forecast.daily_bars import validate_daily_bars
from financial_ai.ml.market_forecast.features import (
    FEATURE_COLUMNS,
    build_market_features,
)
from financial_ai.ml.market_forecast.model_artifact import (
    LoadedMarketForecastModel,
    predict_volatility,
)
from financial_ai.schemas import MarketHistoryRead

FORECAST_HORIZON_TRADING_DAYS = 20


class InsufficientForecastHistoryError(ValueError):
    """Raised when cached history cannot produce a complete forecast input."""


@dataclass(frozen=True)
class MarketForecastInput:
    """Latest observation date and ordered model features for one instrument."""

    observed_on: date
    features: pd.DataFrame


@dataclass(frozen=True)
class MarketVolatilityForecast:
    """Original-scale volatility forecast with model and observation context."""

    symbol: str
    observed_on: date
    horizon_trading_days: int
    predicted_annualized_volatility: float
    model_version: str


def history_to_daily_bars(history: MarketHistoryRead) -> pd.DataFrame:
    """Convert cached market history into the validated daily-bar contract."""
    rows = [
        {
            "symbol": history.instrument.symbol,
            "observed_on": point.observed_on,
            "open": point.open,
            "high": point.high,
            "low": point.low,
            "close": point.close,
            "adjusted_close": point.adjusted_close,
            "volume": point.volume,
        }
        for point in history.points
    ]
    frame = pd.DataFrame(rows)
    return validate_daily_bars(frame)


def build_latest_forecast_input(history: MarketHistoryRead) -> MarketForecastInput:
    """Build the latest complete model input from cached daily history."""
    daily_bars = history_to_daily_bars(history)
    features = build_market_features(daily_bars)
    if features.empty:
        raise InsufficientForecastHistoryError("Market history has no complete feature row")
    latest_row = features.iloc[-1]
    latest_features = features.tail(1).loc[:, FEATURE_COLUMNS].reset_index(drop=True)

    return MarketForecastInput(
        observed_on=latest_row["observed_on"].date(),
        features=latest_features,
    )


def forecast_volatility(
    history: MarketHistoryRead,
    loaded_model: LoadedMarketForecastModel,
) -> MarketVolatilityForecast:
    """Forecast volatility from validated cached history and a verified model."""
    forecast_input = build_latest_forecast_input(history)
    predicted_volatility = predict_volatility(loaded_model, forecast_input.features)
    return MarketVolatilityForecast(
        symbol=history.instrument.symbol.upper(),
        observed_on=forecast_input.observed_on,
        horizon_trading_days=FORECAST_HORIZON_TRADING_DAYS,
        predicted_annualized_volatility=predicted_volatility,
        model_version=loaded_model.metadata.model_version,
    )
