"""XGBoost model selection for market volatility forecasts."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

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
from financial_ai.ml.market_forecast.modeling.model_selection import TemporalValidationFold


@dataclass(frozen=True)
class XGBoostConfig:
    """Controlled tree and regularization settings for one candidate."""

    n_estimators: int = 2_000
    learning_rate: float = 0.03
    max_depth: int = 3
    min_child_weight: float = 10.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int | None = 75


DEFAULT_XGBOOST_CONFIG = XGBoostConfig()


@dataclass(frozen=True)
class XGBoostCandidate:
    """Named hyperparameter configuration evaluated across temporal folds."""

    name: str
    config: XGBoostConfig


DEFAULT_XGBOOST_CANDIDATES = (
    XGBoostCandidate(
        name="shallow",
        config=replace(DEFAULT_XGBOOST_CONFIG, max_depth=2, min_child_weight=20.0),
    ),
    XGBoostCandidate(name="balanced", config=DEFAULT_XGBOOST_CONFIG),
    XGBoostCandidate(
        name="flexible",
        config=replace(DEFAULT_XGBOOST_CONFIG, max_depth=4, min_child_weight=5.0),
    ),
)


@dataclass(frozen=True)
class BoostingFoldEvaluation:
    """Original-scale validation metrics and fitted boosting rounds."""

    validation_year: int
    boosting_rounds: int
    metrics: VolatilityForecastMetrics


@dataclass(frozen=True)
class BoostingCandidateEvaluation:
    """Fold details and aggregate metrics for one configuration."""

    candidate: XGBoostCandidate
    folds: tuple[BoostingFoldEvaluation, ...]
    mean_metrics: VolatilityForecastMetrics
    median_boosting_rounds: int


def evaluate_xgboost_fold(
    fold: TemporalValidationFold,
    config: XGBoostConfig = DEFAULT_XGBOOST_CONFIG,
) -> BoostingFoldEvaluation:
    """Fit one candidate on a temporal fold and evaluate original-scale volatility."""
    required_columns = {TARGET_COLUMN, *FEATURE_COLUMNS}
    for period_name, frame in (("train", fold.train), ("validation", fold.validation)):
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise ValueError(f"XGBoost {period_name} data has missing columns: {names}")

    train_features = fold.train.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    validation_features = fold.validation.loc[:, FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    train_targets = pd.to_numeric(fold.train[TARGET_COLUMN], errors="coerce").to_numpy()
    validation_targets = pd.to_numeric(fold.validation[TARGET_COLUMN], errors="coerce").to_numpy()
    numeric_arrays = (
        train_features.to_numpy(),
        validation_features.to_numpy(),
        train_targets,
        validation_targets,
    )
    if not all(np.isfinite(values).all() for values in numeric_arrays):
        raise ValueError("XGBoost features and targets must be finite")
    if (train_targets <= 0).any() or (validation_targets <= 0).any():
        raise ValueError("XGBoost targets must be positive")

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        **config.__dict__,
    )
    log_train_targets = np.log(train_targets)
    log_validation_targets = np.log(validation_targets)
    model.fit(
        train_features,
        log_train_targets,
        eval_set=[(validation_features, log_validation_targets)],
        verbose=False,
    )
    log_predictions = model.predict(validation_features)
    predicted_values = np.exp(log_predictions)
    return BoostingFoldEvaluation(
        validation_year=fold.validation_year,
        boosting_rounds=model.best_iteration + 1,
        metrics=calculate_volatility_forecast_metrics(validation_targets, predicted_values),
    )


def evaluate_xgboost_candidate(
    candidate: XGBoostCandidate,
    folds: tuple[TemporalValidationFold, ...],
) -> BoostingCandidateEvaluation:
    """Evaluate one candidate across all declared temporal folds."""
    if not candidate.name.strip():
        raise ValueError("XGBoost candidate name cannot be empty")
    if not folds:
        raise ValueError("XGBoost candidate evaluation requires temporal folds")
    fold_evaluations = tuple(evaluate_xgboost_fold(fold, candidate.config) for fold in folds)
    mean_metrics = VolatilityForecastMetrics(
        mae=float(np.mean([result.metrics.mae for result in fold_evaluations])),
        rmse=float(np.mean([result.metrics.rmse for result in fold_evaluations])),
        qlike=float(np.mean([result.metrics.qlike for result in fold_evaluations])),
    )
    median_boosting_rounds = int(
        round(np.median([result.boosting_rounds for result in fold_evaluations]))
    )
    return BoostingCandidateEvaluation(
        candidate=candidate,
        folds=fold_evaluations,
        mean_metrics=mean_metrics,
        median_boosting_rounds=median_boosting_rounds,
    )


def compare_xgboost_candidates(
    folds: tuple[TemporalValidationFold, ...],
    candidates: tuple[XGBoostCandidate, ...] = DEFAULT_XGBOOST_CANDIDATES,
) -> tuple[BoostingCandidateEvaluation, ...]:
    """Evaluate declared candidates and rank them by mean fold MAE."""
    if not candidates:
        raise ValueError("XGBoost comparison requires candidates")
    candidate_names = [candidate.name.strip() for candidate in candidates]
    if any(not name for name in candidate_names) or len(set(candidate_names)) != len(
        candidate_names
    ):
        raise ValueError("XGBoost candidate names must be non-empty and unique")

    candidate_evaluations = tuple(
        evaluate_xgboost_candidate(candidate, folds) for candidate in candidates
    )
    return tuple(
        sorted(
            candidate_evaluations,
            key=lambda evaluation: evaluation.mean_metrics.mae,
        )
    )


def build_xgboost_validation_predictions(
    dataset: pd.DataFrame,
    *,
    config: XGBoostConfig,
    boosting_rounds: int,
) -> pd.DataFrame:
    """Fit a selected fixed-round candidate and predict outer validation volatility."""
    if boosting_rounds < 1:
        raise ValueError("XGBoost boosting rounds must be positive")

    required_columns = {
        "symbol",
        "observed_on",
        TARGET_COLUMN,
        SPLIT_COLUMN,
        *FEATURE_COLUMNS,
    }
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Market model dataset has missing columns: {names}")

    train_rows = dataset.loc[dataset[SPLIT_COLUMN] == "train"]
    validation_rows = dataset.loc[dataset[SPLIT_COLUMN] == "validation"]
    if train_rows.empty or validation_rows.empty:
        raise ValueError("Market model dataset requires train and validation rows")

    train_features = train_rows.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    validation_features = validation_rows.loc[:, FEATURE_COLUMNS].apply(
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
        raise ValueError("XGBoost features and targets must be finite")
    if (train_targets <= 0).any() or (validation_targets <= 0).any():
        raise ValueError("XGBoost targets must be positive")

    model_configuration = config.__dict__.copy()
    model_configuration["n_estimators"] = boosting_rounds
    model_configuration.pop("early_stopping_rounds", None)
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        **model_configuration,
    )
    model.fit(train_features, np.log(train_targets))
    predicted_values = np.exp(model.predict(validation_features))

    result = validation_rows.loc[:, ["symbol", "observed_on", TARGET_COLUMN]].copy()
    result.loc[:, TARGET_COLUMN] = validation_targets
    result[PREDICTED_VOLATILITY_COLUMN] = predicted_values
    return result.loc[:, FORECAST_OUTPUT_COLUMNS].reset_index(drop=True)


def evaluate_xgboost_validation(
    dataset: pd.DataFrame,
    *,
    config: XGBoostConfig,
    boosting_rounds: int,
) -> VolatilityForecastMetrics:
    """Evaluate a selected XGBoost candidate on outer validation data."""
    predictions = build_xgboost_validation_predictions(
        dataset,
        config=config,
        boosting_rounds=boosting_rounds,
    )
    return calculate_volatility_forecast_metrics(
        predictions[TARGET_COLUMN],
        predictions[PREDICTED_VOLATILITY_COLUMN],
    )
