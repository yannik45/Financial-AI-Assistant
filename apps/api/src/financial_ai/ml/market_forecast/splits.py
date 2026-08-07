"""Chronological dataset splits for market forecast evaluation."""

from datetime import date

import pandas as pd

DEFAULT_VALIDATION_START = date(2022, 1, 1)
DEFAULT_TEST_START = date(2024, 1, 1)
DEFAULT_PURGE_TRADING_DAYS = 20
SPLIT_COLUMN = "split"


def assign_chronological_splits(
    frame: pd.DataFrame,
    *,
    validation_start: date = DEFAULT_VALIDATION_START,
    test_start: date = DEFAULT_TEST_START,
    purge_trading_days: int = DEFAULT_PURGE_TRADING_DAYS,
) -> pd.DataFrame:
    """Assign train, validation, and test periods with purged boundaries."""
    if validation_start >= test_start:
        raise ValueError("validation_start must be before test_start")
    if purge_trading_days < 1:
        raise ValueError("purge_trading_days must be at least 1")

    required_columns = {"symbol", "observed_on"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Forecast dataset has missing columns: {names}")

    result = frame.copy(deep=True)
    result["observed_on"] = pd.to_datetime(result["observed_on"], errors="coerce")
    if result["observed_on"].isna().any():
        raise ValueError("Forecast dataset contains invalid observation dates")

    trading_dates = result["observed_on"].drop_duplicates().sort_values()
    validation_boundary = pd.Timestamp(validation_start)
    test_boundary = pd.Timestamp(test_start)

    def dates_to_purge(boundary: pd.Timestamp, period_start: pd.Timestamp | None = None):
        eligible_dates = trading_dates[trading_dates < boundary]
        if period_start is not None:
            eligible_dates = eligible_dates[eligible_dates >= period_start]
        if len(eligible_dates) < purge_trading_days:
            raise ValueError("Not enough trading dates for the configured purge period")
        return eligible_dates.iloc[-purge_trading_days:]

    train_purge_dates = dates_to_purge(validation_boundary)
    validation_purge_dates = dates_to_purge(test_boundary, validation_boundary)

    result[SPLIT_COLUMN] = "test"
    result.loc[result["observed_on"] < test_boundary, SPLIT_COLUMN] = "validation"
    result.loc[result["observed_on"] < validation_boundary, SPLIT_COLUMN] = "train"

    purge_dates = pd.concat([train_purge_dates, validation_purge_dates])
    result = result.loc[~result["observed_on"].isin(purge_dates)]
    if set(result[SPLIT_COLUMN]) != {"train", "validation", "test"}:
        raise ValueError("Chronological split configuration leaves an empty period")

    return result.sort_values(["symbol", "observed_on"]).reset_index(drop=True)
