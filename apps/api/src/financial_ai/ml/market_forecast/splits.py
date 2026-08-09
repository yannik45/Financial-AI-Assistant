"""Chronological dataset splits for market forecast evaluation."""

from datetime import date

import pandas as pd

DEFAULT_VALIDATION_START = date(2022, 1, 1)
DEFAULT_TEST_START = date(2024, 1, 1)
DEFAULT_PURGE_TRADING_DAYS = 20
SPLIT_COLUMN = "split"


def identify_purge_rows(
    frame: pd.DataFrame,
    *,
    boundary: pd.Timestamp,
    purge_trading_days: int,
    period_start: pd.Timestamp | None = None,
) -> pd.Series:
    """Identify each symbol's final observations before a temporal boundary."""
    eligible_mask = frame["observed_on"] < boundary
    if period_start is not None:
        eligible_mask &= frame["observed_on"] >= period_start
    eligible = (
        frame.loc[eligible_mask, ["symbol", "observed_on"]]
        .drop_duplicates()
        .sort_values(["symbol", "observed_on"])
    )
    observations_per_symbol = eligible.groupby("symbol", sort=False).size()
    if observations_per_symbol.empty or observations_per_symbol.lt(purge_trading_days).any():
        raise ValueError("Not enough symbol observations for the configured purge period")
    purge_pairs = eligible.groupby("symbol", sort=False).tail(purge_trading_days)
    purge_index = pd.MultiIndex.from_frame(purge_pairs)
    row_index = pd.MultiIndex.from_frame(frame.loc[:, ["symbol", "observed_on"]])
    return pd.Series(row_index.isin(purge_index), index=frame.index)


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

    validation_boundary = pd.Timestamp(validation_start)
    test_boundary = pd.Timestamp(test_start)
    train_purge_mask = identify_purge_rows(
        result,
        boundary=validation_boundary,
        purge_trading_days=purge_trading_days,
    )
    validation_purge_mask = identify_purge_rows(
        result,
        boundary=test_boundary,
        period_start=validation_boundary,
        purge_trading_days=purge_trading_days,
    )

    result[SPLIT_COLUMN] = "test"
    result.loc[result["observed_on"] < test_boundary, SPLIT_COLUMN] = "validation"
    result.loc[result["observed_on"] < validation_boundary, SPLIT_COLUMN] = "train"

    result = result.loc[~(train_purge_mask | validation_purge_mask)]
    if set(result[SPLIT_COLUMN]) != {"train", "validation", "test"}:
        raise ValueError("Chronological split configuration leaves an empty period")

    return result.sort_values(["symbol", "observed_on"]).reset_index(drop=True)
