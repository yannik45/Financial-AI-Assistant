import pandas as pd
import pytest
from financial_ai.ml.market_forecast.data.targets import TARGET_COLUMN
from financial_ai.ml.market_forecast.evaluation.final_test_report import (
    build_final_test_report,
    prepare_final_test_dataset,
    summarize_test_predictions,
)
from financial_ai.ml.market_forecast.modeling.boosting import XGBoostConfig


def test_final_test_dataset_combines_development_splits_without_modifying_input():
    dataset = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "value": [1, 2, 3],
        }
    )
    expected = dataset.copy(deep=True)

    result = prepare_final_test_dataset(dataset)

    assert result["split"].tolist() == ["train", "train", "validation"]
    pd.testing.assert_frame_equal(dataset, expected)


def test_final_test_summary_reports_metrics_and_signed_bias():
    predictions = pd.DataFrame(
        {
            TARGET_COLUMN: [0.1, 0.2],
            "predicted_volatility": [0.12, 0.18],
        }
    )

    result = summarize_test_predictions(predictions)

    assert result["mae"] == pytest.approx(0.02)
    assert result["rmse"] == pytest.approx(0.02)
    assert result["bias"] == pytest.approx(0.0)


def test_final_test_report_refits_frozen_models_and_reports_predeclared_years(monkeypatch):
    dataset = pd.DataFrame(
        {
            "split": ["train", "validation", "test", "test"],
            "observed_on": pd.to_datetime(["2021-01-04", "2023-01-03", "2024-01-02", "2025-01-02"]),
            "symbol": ["AAPL"] * 4,
            TARGET_COLUMN: [0.1, 0.2, 0.3, 0.4],
        }
    )
    daily_bars = pd.DataFrame({"observed_on": pd.to_datetime(["2024-01-02", "2025-01-02"])})
    captured = {}

    def predictions(received_dataset, value):
        captured.setdefault("fitting_rows", []).append(
            len(received_dataset.loc[received_dataset["split"] == "train"])
        )
        rows = received_dataset.loc[
            received_dataset["split"] == "validation",
            ["symbol", "observed_on", TARGET_COLUMN],
        ].copy()
        rows["predicted_volatility"] = value
        return rows

    monkeypatch.setattr(
        "financial_ai.ml.market_forecast.evaluation.final_test_report.build_ewma_validation_predictions",
        lambda received_dataset, bars: predictions(received_dataset, 0.25),
    )
    monkeypatch.setattr(
        "financial_ai.ml.market_forecast.evaluation.final_test_report.build_ridge_validation_predictions",
        lambda received_dataset: predictions(received_dataset, 0.30),
    )

    def fake_xgboost(received_dataset, *, config, boosting_rounds):
        captured["config"] = config
        captured["boosting_rounds"] = boosting_rounds
        return predictions(received_dataset, 0.35)

    monkeypatch.setattr(
        "financial_ai.ml.market_forecast.evaluation.final_test_report.build_xgboost_validation_predictions",
        fake_xgboost,
    )
    dataset_metadata = {
        "dataset_version": "dataset-v1",
        "sha256": "dataset-checksum",
        "source_snapshot_version": "snapshot-v1",
        "source_snapshot_sha256": "snapshot-checksum",
        "splits": {"test": {"row_count": 2}},
    }
    snapshot_metadata = {
        "snapshot_version": "snapshot-v1",
        "sha256": "snapshot-checksum",
        "provider": "test",
        "feed": "test",
    }
    validation_report = {
        "dataset": {"version": "dataset-v1", "sha256": "dataset-checksum"},
        "model_selection": {
            "candidate": "flexible",
            "configuration": XGBoostConfig(max_depth=4).__dict__,
            "boosting_rounds": 144,
        },
    }

    report = build_final_test_report(
        dataset,
        daily_bars,
        dataset_metadata=dataset_metadata,
        snapshot_metadata=snapshot_metadata,
        validation_report=validation_report,
        validation_version="validation-v1",
        validation_sha256="validation-checksum",
    )

    assert captured["fitting_rows"] == [2, 2, 2]
    assert captured["config"].max_depth == 4
    assert captured["boosting_rounds"] == 144
    assert report["dataset"]["fitting_rows"] == 2
    assert set(report["by_year"]) == {"2024", "2025"}
    assert report["further_tuning_allowed"] is False
