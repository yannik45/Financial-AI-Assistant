"""Regularized linear baseline for market volatility forecasts."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from financial_ai.ml.market_forecast.data.features import FEATURE_COLUMNS
from financial_ai.ml.market_forecast.data.splits import SPLIT_COLUMN
from financial_ai.ml.market_forecast.data.targets import TARGET_COLUMN
from financial_ai.ml.market_forecast.evaluation.evaluation import (
    VolatilityForecastMetrics,
    calculate_volatility_forecast_metrics,
)
from financial_ai.ml.market_forecast.modeling.baselines import (
    FORECAST_OUTPUT_COLUMNS,
    PREDICTED_VOLATILITY_COLUMN,
)
from financial_ai.ml.market_forecast.modeling.ewma import (
    DEFAULT_EWMA_DECAY,
    DEFAULT_EWMA_MIN_OBSERVATIONS,
    EWMA_VOLATILITY_COLUMN,
    build_ewma_volatility,
)

DEFAULT_RIDGE_ALPHA = 1.0


def build_ridge_validation_predictions(
    dataset: pd.DataFrame,
    *,
    alpha: float = DEFAULT_RIDGE_ALPHA,
) -> pd.DataFrame:
    """Fit a pooled Ridge model and predict validation volatility."""
    return _build_ridge_validation_predictions(
        dataset,
        feature_columns=FEATURE_COLUMNS,
        alpha=alpha,
    )


def _build_ridge_validation_predictions(
    dataset: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    alpha: float,
) -> pd.DataFrame:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive")

    required_columns = {
        "symbol",
        "observed_on",
        TARGET_COLUMN,
        SPLIT_COLUMN,
        *feature_columns,
    }
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Market model dataset has missing columns: {names}")

    train_rows = dataset.loc[dataset[SPLIT_COLUMN] == "train"]
    validation_rows = dataset.loc[dataset[SPLIT_COLUMN] == "validation"]
    if train_rows.empty or validation_rows.empty:
        raise ValueError("Market model dataset requires train and validation rows")

    train_features = train_rows.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    validation_features = validation_rows.loc[:, feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    train_targets = pd.to_numeric(train_rows[TARGET_COLUMN], errors="coerce").to_numpy()
    validation_targets = pd.to_numeric(validation_rows[TARGET_COLUMN], errors="coerce").to_numpy()

    numeric_arrays = (
        train_features.to_numpy(),
        validation_features.to_numpy(),
        train_targets,
        validation_targets,
    )
    if not all(np.isfinite(values).all() for values in numeric_arrays):
        raise ValueError("Ridge features and targets must be finite")
    if (train_targets <= 0).any() or (validation_targets <= 0).any():
        raise ValueError("Ridge targets must be positive")

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    log_train_targets = np.log(train_targets)
    model.fit(train_features, log_train_targets)
    log_predictions = model.predict(validation_features)
    predicted_values = np.exp(log_predictions)
    result = validation_rows.loc[:, ["symbol", "observed_on", TARGET_COLUMN]].copy()
    result.loc[:, TARGET_COLUMN] = validation_targets
    result[PREDICTED_VOLATILITY_COLUMN] = predicted_values
    return result.loc[:, FORECAST_OUTPUT_COLUMNS].reset_index(drop=True)


def evaluate_ridge_validation(
    dataset: pd.DataFrame,
    *,
    alpha: float = DEFAULT_RIDGE_ALPHA,
) -> VolatilityForecastMetrics:
    """Evaluate the pooled Ridge forecast on validation data."""
    predictions = build_ridge_validation_predictions(dataset, alpha=alpha)
    return calculate_volatility_forecast_metrics(
        predictions[TARGET_COLUMN],
        predictions[PREDICTED_VOLATILITY_COLUMN],
    )


def build_ridge_ewma_validation_predictions(
    dataset: pd.DataFrame,
    daily_bars: pd.DataFrame,
    *,
    alpha: float = DEFAULT_RIDGE_ALPHA,
    decay: float = DEFAULT_EWMA_DECAY,
    min_observations: int = DEFAULT_EWMA_MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Fit Ridge with the feature contract extended by backward-looking EWMA."""
    if EWMA_VOLATILITY_COLUMN in dataset.columns:
        raise ValueError(f"Market model dataset already contains {EWMA_VOLATILITY_COLUMN}")

    required_columns = {"symbol", "observed_on", SPLIT_COLUMN}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Market model dataset has missing columns: {names}")

    model_rows = dataset.loc[dataset[SPLIT_COLUMN].isin(["train", "validation"])].copy()
    model_rows["observed_on"] = pd.to_datetime(model_rows["observed_on"], errors="coerce")
    if model_rows["observed_on"].isna().any():
        raise ValueError("Market model dataset contains invalid observation dates")

    ewma = build_ewma_volatility(
        daily_bars,
        decay=decay,
        min_observations=min_observations,
    )
    augmented = model_rows.merge(
        ewma,
        on=["symbol", "observed_on"],
        how="left",
        validate="one_to_one",
    )
    if augmented[EWMA_VOLATILITY_COLUMN].isna().any():
        raise ValueError("Train or validation rows are missing EWMA features")

    return _build_ridge_validation_predictions(
        augmented,
        feature_columns=(*FEATURE_COLUMNS, EWMA_VOLATILITY_COLUMN),
        alpha=alpha,
    )


def evaluate_ridge_ewma_validation(
    dataset: pd.DataFrame,
    daily_bars: pd.DataFrame,
    *,
    alpha: float = DEFAULT_RIDGE_ALPHA,
    decay: float = DEFAULT_EWMA_DECAY,
    min_observations: int = DEFAULT_EWMA_MIN_OBSERVATIONS,
) -> VolatilityForecastMetrics:
    """Evaluate Ridge with EWMA added to the validation feature contract."""
    predictions = build_ridge_ewma_validation_predictions(
        dataset,
        daily_bars,
        alpha=alpha,
        decay=decay,
        min_observations=min_observations,
    )
    return calculate_volatility_forecast_metrics(
        predictions[TARGET_COLUMN],
        predictions[PREDICTED_VOLATILITY_COLUMN],
    )
