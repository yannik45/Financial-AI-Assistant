"""Time-aware model-selection folds for market volatility forecasts."""

from dataclasses import dataclass

import pandas as pd

from financial_ai.ml.market_forecast.splits import (
    DEFAULT_PURGE_TRADING_DAYS,
    SPLIT_COLUMN,
    identify_purge_rows,
)

DEFAULT_INNER_VALIDATION_YEARS = (2019, 2020, 2021)


@dataclass(frozen=True)
class TemporalValidationFold:
    """Expanding training data and its following validation year."""

    validation_year: int
    train: pd.DataFrame
    validation: pd.DataFrame


def build_expanding_training_folds(
    dataset: pd.DataFrame,
    *,
    validation_years: tuple[int, ...] = DEFAULT_INNER_VALIDATION_YEARS,
    purge_trading_days: int = DEFAULT_PURGE_TRADING_DAYS,
) -> tuple[TemporalValidationFold, ...]:
    """Build purged expanding-window folds from the outer training split."""
    required_columns = {"symbol", "observed_on", SPLIT_COLUMN}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Market model dataset has missing columns: {names}")
    if not validation_years or tuple(sorted(set(validation_years))) != validation_years:
        raise ValueError("Validation years must be unique and strictly increasing")
    if purge_trading_days < 1:
        raise ValueError("Purge trading days must be at least one")

    outer_train = dataset.loc[dataset[SPLIT_COLUMN] == "train"].copy()
    if outer_train.empty:
        raise ValueError("Market model dataset contains no outer training rows")
    outer_train["observed_on"] = pd.to_datetime(outer_train["observed_on"], errors="coerce")
    if outer_train["observed_on"].isna().any():
        raise ValueError("Outer training split contains invalid observation dates")
    folds = []
    for validation_year in validation_years:
        validation_start = pd.Timestamp(year=validation_year, month=1, day=1)
        validation_end = pd.Timestamp(year=validation_year + 1, month=1, day=1)
        validation_mask = outer_train["observed_on"].between(
            validation_start,
            validation_end,
            inclusive="left",
        )
        validation_rows = outer_train.loc[validation_mask].copy()
        purge_mask = identify_purge_rows(
            outer_train,
            boundary=validation_start,
            purge_trading_days=purge_trading_days,
        )
        train_mask = (~purge_mask) & (outer_train["observed_on"] < validation_start)
        train_rows = outer_train.loc[train_mask].copy()
        if train_rows.empty or validation_rows.empty:
            raise ValueError(f"Training fold for {validation_year} contains an empty period")
        folds.append(
            TemporalValidationFold(
                validation_year=validation_year,
                train=train_rows.reset_index(drop=True),
                validation=validation_rows.reset_index(drop=True),
            )
        )
    return tuple(folds)
