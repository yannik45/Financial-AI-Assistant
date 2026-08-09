import pandas as pd
import pytest
from financial_ai.ml.market_forecast.daily_bars import (
    DAILY_BAR_COLUMNS,
    DailyBarValidationError,
    validate_daily_bars,
)


def valid_bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": " msft ",
                "observed_on": "2024-01-03",
                "open": "370.00",
                "high": "376.00",
                "low": "369.00",
                "close": "374.50",
                "adjusted_close": "374.50",
                "volume": "23000000",
            },
            {
                "symbol": "AAPL",
                "observed_on": "2024-01-02",
                "open": "186.00",
                "high": "188.00",
                "low": "183.50",
                "close": "185.64",
                "adjusted_close": "185.64",
                "volume": "82488700",
            },
        ]
    )


def test_daily_bars_are_normalized_without_mutating_the_input():
    source = valid_bars()
    original = source.copy(deep=True)

    result = validate_daily_bars(source)

    pd.testing.assert_frame_equal(source, original)
    assert tuple(result.columns) == DAILY_BAR_COLUMNS
    assert result["symbol"].tolist() == ["AAPL", "MSFT"]
    assert pd.api.types.is_datetime64_any_dtype(result["observed_on"])
    assert all(pd.api.types.is_numeric_dtype(result[column]) for column in DAILY_BAR_COLUMNS[2:])


def test_daily_bars_reject_missing_columns():
    frame = valid_bars().drop(columns="volume")

    with pytest.raises(DailyBarValidationError, match="missing columns"):
        validate_daily_bars(frame)


def test_daily_bars_reject_duplicate_symbol_dates_after_normalization():
    frame = pd.concat([valid_bars(), valid_bars().iloc[[0]].assign(symbol="MSFT")])

    with pytest.raises(DailyBarValidationError, match="duplicate"):
        validate_daily_bars(frame)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("open", "not-a-number", "numeric"),
        ("open", "inf", "finite"),
        ("close", "0", "positive"),
        ("volume", "-1", "volume"),
        ("high", "180", "OHLC"),
        ("low", "190", "OHLC"),
    ],
)
def test_daily_bars_reject_invalid_market_values(column: str, value: str, message: str):
    frame = valid_bars()
    frame.loc[1, column] = value

    with pytest.raises(DailyBarValidationError, match=message):
        validate_daily_bars(frame)
