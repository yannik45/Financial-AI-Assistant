"""Validation contract for daily OHLCV observations used by forecasting experiments."""

import pandas as pd

DAILY_BAR_COLUMNS = (
    "symbol",
    "observed_on",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)


class DailyBarValidationError(ValueError):
    """Raised when raw market observations violate the snapshot contract."""


def validate_daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of valid daily bars.

    The returned frame must contain exactly the columns in ``DAILY_BAR_COLUMNS``,
    use uppercase symbols, contain parsed observation dates and numeric market
    values, and be sorted by symbol and date. Invalid schemas, duplicate
    symbol/date pairs, missing values, impossible OHLC relationships, non-positive
    prices, or negative volume must raise ``DailyBarValidationError``.

    """

    df = frame.copy(deep=True)
    missing = set(DAILY_BAR_COLUMNS).difference(df.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise DailyBarValidationError(f"Daily bars have missing columns: {names}")

    df = df.loc[:, list(DAILY_BAR_COLUMNS)]
    df["symbol"] = df["symbol"].astype("string").str.strip().str.upper()
    if df["symbol"].isna().any() or df["symbol"].eq("").any():
        raise DailyBarValidationError("Daily bars contain missing symbols")

    df["observed_on"] = pd.to_datetime(df["observed_on"], errors="coerce")
    if df["observed_on"].isna().any():
        raise DailyBarValidationError("Daily bars contain invalid or missing observation dates")

    numeric_columns = list(DAILY_BAR_COLUMNS[2:])
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[numeric_columns].isna().any().any():
        raise DailyBarValidationError("Daily bars must contain numeric values without gaps")

    price_columns = ["open", "high", "low", "close", "adjusted_close"]
    if df[price_columns].le(0).any().any():
        raise DailyBarValidationError("Daily-bar prices must be positive")
    if df["volume"].lt(0).any():
        raise DailyBarValidationError("Daily-bar volume must not be negative")

    invalid_ohlc = (
        df["low"].gt(df[["open", "close"]].min(axis=1))
        | df["high"].lt(df[["open", "close"]].max(axis=1))
        | df["low"].gt(df["high"])
    )
    if invalid_ohlc.any():
        raise DailyBarValidationError("Daily bars contain invalid OHLC relationships")

    if df.duplicated(subset=["symbol", "observed_on"]).any():
        raise DailyBarValidationError("Daily bars contain duplicate symbol/date observations")

    return df.sort_values(["symbol", "observed_on"]).reset_index(drop=True)
