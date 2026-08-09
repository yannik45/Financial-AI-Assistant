import pandas as pd
import pytest
from financial_ai.ml.market_forecast.boosting_diagnostics import (
    build_diagnostics_report,
    summarize_forecast_slice,
)
from financial_ai.ml.market_forecast.targets import TARGET_COLUMN


def comparison_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "observed_on": pd.to_datetime(["2022-01-03", "2023-01-03", "2022-01-03", "2023-01-03"]),
            TARGET_COLUMN: [0.10, 0.20, 0.30, 0.40],
            "ewma_prediction": [0.12, 0.18, 0.28, 0.35],
            "ridge_prediction": [0.11, 0.19, 0.29, 0.38],
            "xgboost_prediction": [0.10, 0.21, 0.31, 0.39],
        }
    )


def test_forecast_slice_reports_metrics_and_signed_bias():
    result = summarize_forecast_slice(comparison_rows())

    assert result["row_count"] == 4
    assert result["mean_actual_volatility"] == pytest.approx(0.25)
    assert result["xgboost"]["mae"] == pytest.approx(0.0075)
    assert result["xgboost"]["bias"] == pytest.approx(0.0025)
    assert result["ewma"]["bias"] == pytest.approx(-0.0175)


def test_diagnostics_report_builds_year_symbol_and_fixed_volatility_slices():
    report = build_diagnostics_report(
        comparison_rows(),
        dataset_metadata={"dataset_version": "dataset-v1", "sha256": "checksum"},
        selection_version="inner-cv-v1",
        selection_sha256="selection-checksum",
        candidate_name="flexible",
        boosting_rounds=144,
    )

    assert report["parameter_changes_allowed"] is False
    assert report["test_split_evaluated"] is False
    assert set(report["by_year"]) == {"2022", "2023"}
    assert set(report["by_symbol"]) == {"AAPL", "MSFT"}
    assert set(report["by_volatility_regime"]) == {
        "low_below_15pct",
        "moderate_15_to_30pct",
        "high_at_least_30pct",
    }
