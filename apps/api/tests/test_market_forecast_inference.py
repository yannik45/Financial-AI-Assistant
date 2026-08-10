from datetime import UTC, date, datetime
from decimal import Decimal

import pandas as pd
import pytest
from financial_ai.ml.market_forecast.daily_bars import (
    DAILY_BAR_COLUMNS,
    DailyBarValidationError,
)
from financial_ai.ml.market_forecast.features import FEATURE_COLUMNS, build_market_features
from financial_ai.ml.market_forecast.inference import (
    FORECAST_HORIZON_TRADING_DAYS,
    InsufficientForecastHistoryError,
    build_latest_forecast_input,
    forecast_volatility,
    history_to_daily_bars,
)
from financial_ai.schemas import MarketHistoryRead, MarketInstrumentRead, MarketPriceRead


def market_history(*, missing_open: bool = False) -> MarketHistoryRead:
    instrument = MarketInstrumentRead(
        id="instrument-1",
        provider="alpaca",
        symbol="aapl",
        name="Apple Inc.",
        exchange="NASDAQ",
        currency="USD",
        asset_class="US Equity",
        region="United States",
        is_active=True,
        updated_at=datetime(2026, 8, 10, tzinfo=UTC),
    )
    return MarketHistoryRead(
        instrument=instrument,
        source="alpaca",
        retrieved_at=datetime(2026, 8, 10, tzinfo=UTC),
        points=[
            MarketPriceRead(
                observed_on=date(2026, 8, 6),
                open=None if missing_open else Decimal("201.00"),
                high=Decimal("204.00"),
                low=Decimal("200.00"),
                close=Decimal("203.00"),
                adjusted_close=Decimal("203.00"),
                volume=Decimal("42000000"),
            ),
            MarketPriceRead(
                observed_on=date(2026, 8, 7),
                open=Decimal("203.00"),
                high=Decimal("206.00"),
                low=Decimal("202.00"),
                close=Decimal("205.00"),
                adjusted_close=Decimal("205.00"),
                volume=Decimal("45000000"),
            ),
        ],
    )


def forecast_history(periods: int) -> MarketHistoryRead:
    history = market_history()
    dates = pd.bdate_range("2026-05-01", periods=periods)
    history.points = [
        MarketPriceRead(
            observed_on=observed_on.date(),
            open=Decimal(str(100 + index)),
            high=Decimal(str(102 + index)),
            low=Decimal(str(99 + index)),
            close=Decimal(str(101 + index)),
            adjusted_close=Decimal(str(101 + index)),
            volume=Decimal(str(1_000_000 + index * 1_000)),
        )
        for index, observed_on in enumerate(dates)
    ]
    return history


def test_market_history_is_converted_to_validated_daily_bars():
    history = market_history()

    result = history_to_daily_bars(history)

    assert tuple(result.columns) == DAILY_BAR_COLUMNS
    assert result["symbol"].tolist() == ["AAPL", "AAPL"]
    assert result["observed_on"].tolist() == [
        pd.Timestamp("2026-08-06"),
        pd.Timestamp("2026-08-07"),
    ]
    assert result["adjusted_close"].tolist() == [203.0, 205.0]
    assert result["volume"].tolist() == [42_000_000.0, 45_000_000.0]


def test_market_history_rejects_incomplete_ohlcv_bars():
    with pytest.raises(DailyBarValidationError, match="without gaps"):
        history_to_daily_bars(market_history(missing_open=True))


def test_latest_forecast_input_uses_latest_complete_feature_row():
    history = forecast_history(65)
    expected_features = build_market_features(history_to_daily_bars(history)).iloc[-1]

    result = build_latest_forecast_input(history)

    assert result.observed_on == history.points[-1].observed_on
    assert result.features.shape == (1, len(FEATURE_COLUMNS))
    assert tuple(result.features.columns) == FEATURE_COLUMNS
    assert result.features.iloc[0].tolist() == pytest.approx(
        expected_features.loc[list(FEATURE_COLUMNS)].tolist()
    )


def test_latest_forecast_input_rejects_history_without_complete_features():
    with pytest.raises(InsufficientForecastHistoryError, match="complete feature row"):
        build_latest_forecast_input(forecast_history(60))


def test_volatility_forecast_includes_observation_and_model_context(monkeypatch):
    history = forecast_history(65)
    loaded_model = type(
        "LoadedModelStub",
        (),
        {"metadata": type("MetadataStub", (), {"model_version": "model-v1"})()},
    )()
    monkeypatch.setattr(
        "financial_ai.ml.market_forecast.inference.predict_volatility",
        lambda model, features: 0.237,
    )

    result = forecast_volatility(history, loaded_model)

    assert result.symbol == "AAPL"
    assert result.observed_on == history.points[-1].observed_on
    assert result.horizon_trading_days == FORECAST_HORIZON_TRADING_DAYS
    assert result.predicted_annualized_volatility == pytest.approx(0.237)
    assert result.model_version == "model-v1"
