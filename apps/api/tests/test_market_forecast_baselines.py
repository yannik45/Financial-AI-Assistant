import numpy as np
import pandas as pd
import pytest
from financial_ai.ml.market_forecast.data.targets import TARGET_COLUMN
from financial_ai.ml.market_forecast.evaluation.evaluation import (
    calculate_volatility_forecast_metrics,
)
from financial_ai.ml.market_forecast.modeling.baselines import (
    PREDICTED_VOLATILITY_COLUMN,
    build_constant_validation_predictions,
    build_ewma_validation_predictions,
    build_persistence_validation_predictions,
    evaluate_ewma_validation,
    evaluate_persistence_validation,
)
from financial_ai.ml.market_forecast.modeling.ewma import (
    EWMA_VOLATILITY_COLUMN,
    build_ewma_volatility,
)


def model_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "observed_on": pd.to_datetime(["2021-12-30", "2022-01-03", "2022-01-03", "2024-01-02"]),
            "realized_volatility_20d": [0.18, 0.20, 0.25, 0.30],
            TARGET_COLUMN: [0.19, 0.22, 0.28, 0.35],
            "split": ["train", "validation", "validation", "test"],
        }
    )


def ewma_daily_bars() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    rows = []
    for symbol, closes in (
        ("AAPL", [100.0, 102.0, 101.0, 104.0]),
        ("MSFT", [200.0, 198.0, 202.0, 201.0]),
    ):
        prices = np.asarray(closes)
        rows.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "observed_on": dates,
                    "open": prices,
                    "high": prices * 1.01,
                    "low": prices * 0.99,
                    "close": prices,
                    "adjusted_close": prices,
                    "volume": 1_000_000,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_persistence_predictions_use_only_validation_rows_and_trailing_volatility():
    result = build_persistence_validation_predictions(model_dataset())

    assert result["symbol"].tolist() == ["AAPL", "MSFT"]
    assert result[TARGET_COLUMN].tolist() == [0.22, 0.28]
    assert result[PREDICTED_VOLATILITY_COLUMN].tolist() == [0.20, 0.25]
    assert "split" not in result.columns


def test_persistence_predictions_do_not_modify_model_dataset():
    source = model_dataset()
    expected = source.copy(deep=True)

    build_persistence_validation_predictions(source)

    pd.testing.assert_frame_equal(source, expected)


def test_persistence_predictions_reject_missing_columns():
    source = model_dataset().drop(columns=["realized_volatility_20d"])

    with pytest.raises(ValueError, match="missing columns"):
        build_persistence_validation_predictions(source)


def test_persistence_predictions_reject_dataset_without_validation_rows():
    source = model_dataset()
    source["split"] = "train"

    with pytest.raises(ValueError, match="validation rows"):
        build_persistence_validation_predictions(source)


@pytest.mark.parametrize("invalid_value", [0.0, -0.1, float("nan"), float("inf")])
def test_persistence_predictions_reject_invalid_volatility(invalid_value):
    source = model_dataset()
    source.loc[source["split"] == "validation", "realized_volatility_20d"] = invalid_value

    with pytest.raises(ValueError, match="finite|positive"):
        build_persistence_validation_predictions(source)


def test_persistence_validation_evaluation_uses_prediction_contract():
    result = evaluate_persistence_validation(model_dataset())

    expected = calculate_volatility_forecast_metrics([0.22, 0.28], [0.20, 0.25])
    assert result == expected


@pytest.mark.parametrize(
    ("strategy", "expected_prediction"),
    [("mean", 0.30), ("median", 0.20)],
)
def test_constant_predictions_fit_only_train_targets(strategy, expected_prediction):
    source = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "NVDA", "AAPL", "MSFT", "NVDA"],
            "observed_on": pd.date_range("2021-12-27", periods=6, freq="B"),
            TARGET_COLUMN: [0.10, 0.20, 0.60, 0.80, 0.90, 1.00],
            "split": ["train", "train", "train", "validation", "validation", "test"],
        }
    )

    result = build_constant_validation_predictions(source, strategy)

    assert result[PREDICTED_VOLATILITY_COLUMN].tolist() == pytest.approx(
        [expected_prediction, expected_prediction]
    )
    assert result[TARGET_COLUMN].tolist() == [0.80, 0.90]


def test_ewma_predictions_align_complete_price_history_with_validation_targets():
    bars = ewma_daily_bars()
    ewma = build_ewma_volatility(bars, decay=0.5, min_observations=2)
    expected = ewma.tail(2).reset_index(drop=True)
    dataset = expected.loc[:, ["symbol", "observed_on"]].copy()
    dataset[TARGET_COLUMN] = [0.30, 0.40]
    dataset["split"] = "validation"

    result = build_ewma_validation_predictions(
        dataset,
        bars,
        decay=0.5,
        min_observations=2,
    )

    assert result[PREDICTED_VOLATILITY_COLUMN].tolist() == pytest.approx(
        expected[EWMA_VOLATILITY_COLUMN].tolist()
    )
    assert result[TARGET_COLUMN].tolist() == [0.30, 0.40]


def test_ewma_validation_evaluation_uses_aligned_predictions():
    bars = ewma_daily_bars()
    ewma = build_ewma_volatility(bars, decay=0.5, min_observations=2)
    expected = ewma.tail(2).reset_index(drop=True)
    dataset = expected.loc[:, ["symbol", "observed_on"]].copy()
    dataset[TARGET_COLUMN] = [0.30, 0.40]
    dataset["split"] = "validation"

    result = evaluate_ewma_validation(
        dataset,
        bars,
        decay=0.5,
        min_observations=2,
    )

    expected_metrics = calculate_volatility_forecast_metrics(
        dataset[TARGET_COLUMN],
        expected[EWMA_VOLATILITY_COLUMN],
    )
    assert result == expected_metrics
