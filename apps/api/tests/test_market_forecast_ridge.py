import numpy as np
import pandas as pd
import pytest
from financial_ai.ml.market_forecast.data.features import FEATURE_COLUMNS
from financial_ai.ml.market_forecast.data.targets import TARGET_COLUMN
from financial_ai.ml.market_forecast.evaluation.evaluation import (
    calculate_volatility_forecast_metrics,
)
from financial_ai.ml.market_forecast.modeling.baselines import PREDICTED_VOLATILITY_COLUMN
from financial_ai.ml.market_forecast.modeling.ridge import (
    build_ridge_ewma_validation_predictions,
    build_ridge_validation_predictions,
    evaluate_ridge_ewma_validation,
    evaluate_ridge_validation,
)


def model_dataset() -> pd.DataFrame:
    rows = []
    for index in range(12):
        split = "train" if index < 8 else "validation" if index < 10 else "test"
        feature_level = 0.01 * (index + 1)
        row = {
            "symbol": "AAPL" if index % 2 == 0 else "MSFT",
            "observed_on": pd.Timestamp("2021-01-04") + pd.offsets.BDay(index),
            TARGET_COLUMN: float(np.exp(-2.0 + 3.0 * feature_level)),
            "split": split,
        }
        row.update(
            {
                feature: feature_level * (feature_index + 1)
                for feature_index, feature in enumerate(FEATURE_COLUMNS)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def daily_bars() -> pd.DataFrame:
    dates = pd.date_range("2020-12-29", periods=16, freq="B")
    frames = []
    for symbol, phase in (("AAPL", 0.0), ("MSFT", 0.5)):
        closes = 100 * np.exp(np.sin(np.arange(16) / 3 + phase) * 0.04)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "observed_on": dates,
                    "open": closes,
                    "high": closes * 1.01,
                    "low": closes * 0.99,
                    "close": closes,
                    "adjusted_close": closes,
                    "volume": 1_000_000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_ridge_predictions_fit_train_and_predict_only_validation_rows():
    result = build_ridge_validation_predictions(model_dataset())

    assert result["symbol"].tolist() == ["AAPL", "MSFT"]
    assert result[TARGET_COLUMN].gt(0).all()
    assert result[PREDICTED_VOLATILITY_COLUMN].gt(0).all()


def test_ridge_predictions_do_not_depend_on_test_targets():
    source = model_dataset()
    changed = source.copy(deep=True)
    changed.loc[changed["split"] == "test", TARGET_COLUMN] = 1000.0

    original_predictions = build_ridge_validation_predictions(source)
    changed_predictions = build_ridge_validation_predictions(changed)

    pd.testing.assert_frame_equal(original_predictions, changed_predictions)


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf")])
def test_ridge_predictions_reject_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        build_ridge_validation_predictions(model_dataset(), alpha=alpha)


def test_ridge_validation_evaluation_uses_prediction_contract():
    predictions = build_ridge_validation_predictions(model_dataset())

    result = evaluate_ridge_validation(model_dataset())

    expected = calculate_volatility_forecast_metrics(
        predictions[TARGET_COLUMN],
        predictions[PREDICTED_VOLATILITY_COLUMN],
    )
    assert result == expected


def test_ridge_ewma_predictions_extend_features_with_complete_price_history():
    result = build_ridge_ewma_validation_predictions(
        model_dataset(),
        daily_bars(),
        min_observations=2,
    )

    assert result["symbol"].tolist() == ["AAPL", "MSFT"]
    assert result[PREDICTED_VOLATILITY_COLUMN].gt(0).all()


def test_ridge_ewma_validation_evaluation_uses_augmented_predictions():
    predictions = build_ridge_ewma_validation_predictions(
        model_dataset(),
        daily_bars(),
        min_observations=2,
    )

    result = evaluate_ridge_ewma_validation(
        model_dataset(),
        daily_bars(),
        min_observations=2,
    )

    expected = calculate_volatility_forecast_metrics(
        predictions[TARGET_COLUMN],
        predictions[PREDICTED_VOLATILITY_COLUMN],
    )
    assert result == expected
