"""Evaluation metrics for market volatility forecasts."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class VolatilityForecastMetrics:
    """Aggregate errors for annualized volatility forecasts."""

    mae: float
    rmse: float
    qlike: float


def calculate_volatility_forecast_metrics(
    actual_volatility: ArrayLike,
    predicted_volatility: ArrayLike,
) -> VolatilityForecastMetrics:
    """Calculate scale-dependent and volatility-specific forecast errors."""
    actual = np.asarray(actual_volatility, dtype=float)
    predicted = np.asarray(predicted_volatility, dtype=float)
    if actual.size == 0 or predicted.size == 0:
        raise ValueError("Volatility arrays cannot be empty")
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted volatility must have the same shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Volatility values must be finite")
    if (actual <= 0).any() or (predicted <= 0).any():
        raise ValueError("Volatility values must be positive")

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    actual_variance = actual**2
    predicted_variance = predicted**2
    ratio = actual_variance / predicted_variance
    qlike = np.mean(ratio - np.log(ratio) - 1)
    return VolatilityForecastMetrics(
        mae=float(mae),
        rmse=float(rmse),
        qlike=float(qlike),
    )
