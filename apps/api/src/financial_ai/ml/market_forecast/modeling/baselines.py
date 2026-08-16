"""Reference forecasts for market volatility models."""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from financial_ai.ml.market_forecast.data.splits import SPLIT_COLUMN
from financial_ai.ml.market_forecast.data.targets import TARGET_COLUMN
from financial_ai.ml.market_forecast.evaluation.evaluation import (
    VolatilityForecastMetrics,
    calculate_volatility_forecast_metrics,
)
from financial_ai.ml.market_forecast.modeling.ewma import (
    DEFAULT_EWMA_DECAY,
    DEFAULT_EWMA_MIN_OBSERVATIONS,
    EWMA_VOLATILITY_COLUMN,
    build_ewma_volatility,
)

PREDICTED_VOLATILITY_COLUMN = "predicted_volatility"
PERSISTENCE_FEATURE_COLUMN = "realized_volatility_20d"
ConstantStrategy = Literal["mean", "median"]
FORECAST_OUTPUT_COLUMNS = (
    "symbol",
    "observed_on",
    TARGET_COLUMN,
    PREDICTED_VOLATILITY_COLUMN,
)


def build_persistence_validation_predictions(dataset: pd.DataFrame) -> pd.DataFrame:
    """Build validation predictions from trailing 20-day realized volatility."""
    required_columns = {
        "symbol",
        "observed_on",
        PERSISTENCE_FEATURE_COLUMN,
        TARGET_COLUMN,
        SPLIT_COLUMN,
    }
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Market model dataset has missing columns: {names}")

    validation_rows = dataset.loc[dataset[SPLIT_COLUMN] == "validation"]
    if validation_rows.empty:
        raise ValueError("Market model dataset contains no validation rows")

    volatility_columns = [PERSISTENCE_FEATURE_COLUMN, TARGET_COLUMN]
    numeric_values = validation_rows.loc[:, volatility_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(numeric_values.to_numpy()).all():
        raise ValueError("Validation volatility values must be finite")
    if numeric_values.le(0).any().any():
        raise ValueError("Validation volatility values must be positive")

    persistence_df = validation_rows.loc[
        :, ["symbol", "observed_on", TARGET_COLUMN, PERSISTENCE_FEATURE_COLUMN]
    ].copy()
    persistence_df.loc[:, volatility_columns] = numeric_values
    persistence_df = persistence_df.rename(
        columns={PERSISTENCE_FEATURE_COLUMN: PREDICTED_VOLATILITY_COLUMN}
    )
    return persistence_df.loc[:, FORECAST_OUTPUT_COLUMNS].reset_index(drop=True)


def evaluate_persistence_validation(dataset: pd.DataFrame) -> VolatilityForecastMetrics:
    """Evaluate the persistence reference forecast on validation data."""
    predictions = build_persistence_validation_predictions(dataset)
    actual_values = predictions[TARGET_COLUMN]
    predicted_values = predictions[PREDICTED_VOLATILITY_COLUMN]
    return calculate_volatility_forecast_metrics(actual_values, predicted_values)


def build_constant_validation_predictions(
    dataset: pd.DataFrame,
    strategy: ConstantStrategy,
) -> pd.DataFrame:
    """Fit a feature-free constant baseline and predict validation volatility."""
    if strategy not in {"mean", "median"}:
        raise ValueError("Constant baseline strategy must be mean or median")

    required_columns = {"symbol", "observed_on", TARGET_COLUMN, SPLIT_COLUMN}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Market model dataset has missing columns: {names}")

    train_rows = dataset.loc[dataset[SPLIT_COLUMN] == "train"]
    validation_rows = dataset.loc[dataset[SPLIT_COLUMN] == "validation"]
    if train_rows.empty or validation_rows.empty:
        raise ValueError("Market model dataset requires train and validation rows")

    train_targets = pd.to_numeric(train_rows[TARGET_COLUMN], errors="coerce").to_numpy()
    validation_targets = pd.to_numeric(validation_rows[TARGET_COLUMN], errors="coerce").to_numpy()
    if not np.isfinite(train_targets).all() or not np.isfinite(validation_targets).all():
        raise ValueError("Train and validation targets must be finite")
    if (train_targets <= 0).any() or (validation_targets <= 0).any():
        raise ValueError("Train and validation targets must be positive")

    model = DummyRegressor(strategy=strategy)
    train_inputs = np.zeros((train_targets.size, 1))
    validation_inputs = np.zeros((validation_targets.size, 1))
    model.fit(train_inputs, train_targets)
    predicted_values = model.predict(validation_inputs)

    result = validation_rows.loc[:, ["symbol", "observed_on", TARGET_COLUMN]].copy()
    result.loc[:, TARGET_COLUMN] = validation_targets
    result[PREDICTED_VOLATILITY_COLUMN] = predicted_values
    return result.loc[:, FORECAST_OUTPUT_COLUMNS].reset_index(drop=True)


def build_ewma_validation_predictions(
    dataset: pd.DataFrame,
    daily_bars: pd.DataFrame,
    *,
    decay: float = DEFAULT_EWMA_DECAY,
    min_observations: int = DEFAULT_EWMA_MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Align backward-looking EWMA forecasts with validation targets."""
    required_columns = {"symbol", "observed_on", TARGET_COLUMN, SPLIT_COLUMN}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Market model dataset has missing columns: {names}")

    validation_rows = dataset.loc[
        dataset[SPLIT_COLUMN] == "validation",
        ["symbol", "observed_on", TARGET_COLUMN],
    ].copy()
    if validation_rows.empty:
        raise ValueError("Market model dataset contains no validation rows")

    validation_rows["observed_on"] = pd.to_datetime(validation_rows["observed_on"], errors="coerce")
    validation_rows[TARGET_COLUMN] = pd.to_numeric(validation_rows[TARGET_COLUMN], errors="coerce")
    if validation_rows[["observed_on", TARGET_COLUMN]].isna().any().any():
        raise ValueError("Validation rows contain invalid dates or targets")
    if (validation_rows[TARGET_COLUMN] <= 0).any():
        raise ValueError("Validation targets must be positive")

    ewma = build_ewma_volatility(
        daily_bars,
        decay=decay,
        min_observations=min_observations,
    )
    merged = validation_rows.merge(
        ewma,
        on=["symbol", "observed_on"],
        how="left",
        validate="one_to_one",
    )
    if merged[EWMA_VOLATILITY_COLUMN].isna().any():
        raise ValueError("DataFrame is missing EWMA predictions")
    merged = merged.rename(columns={EWMA_VOLATILITY_COLUMN: PREDICTED_VOLATILITY_COLUMN})
    return merged.loc[:, FORECAST_OUTPUT_COLUMNS].reset_index(drop=True)


def evaluate_ewma_validation(
    dataset: pd.DataFrame,
    daily_bars: pd.DataFrame,
    *,
    decay: float = DEFAULT_EWMA_DECAY,
    min_observations: int = DEFAULT_EWMA_MIN_OBSERVATIONS,
) -> VolatilityForecastMetrics:
    """Evaluate the EWMA reference forecast on validation data."""
    predictions = build_ewma_validation_predictions(
        dataset,
        daily_bars,
        decay=decay,
        min_observations=min_observations,
    )
    return calculate_volatility_forecast_metrics(
        predictions[TARGET_COLUMN],
        predictions[PREDICTED_VOLATILITY_COLUMN],
    )
