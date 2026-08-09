"""Descriptive outer-validation diagnostics for volatility forecasts."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from financial_ai.ml.artifact_integrity import normalize_artifact_version
from financial_ai.ml.market_forecast.baselines import (
    PREDICTED_VOLATILITY_COLUMN,
    build_ewma_validation_predictions,
)
from financial_ai.ml.market_forecast.boosting import (
    XGBoostConfig,
    build_xgboost_validation_predictions,
)
from financial_ai.ml.market_forecast.boosting_report import (
    DEFAULT_BOOSTING_SELECTION_DIRECTORY,
    load_boosting_selection_report,
)
from financial_ai.ml.market_forecast.dataset import (
    DEFAULT_DATASET_DIRECTORY,
    load_model_dataset,
)
from financial_ai.ml.market_forecast.evaluation import calculate_volatility_forecast_metrics
from financial_ai.ml.market_forecast.ridge import build_ridge_validation_predictions
from financial_ai.ml.market_forecast.snapshot import (
    DEFAULT_OUTPUT_DIRECTORY as DEFAULT_SNAPSHOT_DIRECTORY,
)
from financial_ai.ml.market_forecast.snapshot import load_market_snapshot
from financial_ai.ml.market_forecast.splits import SPLIT_COLUMN
from financial_ai.ml.market_forecast.targets import TARGET_COLUMN
from financial_ai.ml.market_forecast.validation_report import DEFAULT_EVALUATION_DIRECTORY

BOOSTING_DIAGNOSTICS_SCHEMA_VERSION = "market-volatility-boosting-diagnostics-v1"
FORECAST_COLUMNS = {
    "ewma": "ewma_prediction",
    "ridge": "ridge_prediction",
    "xgboost": "xgboost_prediction",
}


def build_forecast_comparison(
    dataset: pd.DataFrame,
    daily_bars: pd.DataFrame,
    *,
    config: XGBoostConfig,
    boosting_rounds: int,
) -> pd.DataFrame:
    """Align outer-validation targets and model forecasts by instrument and date."""
    validation_rows = dataset.loc[dataset[SPLIT_COLUMN] == "validation"]
    if validation_rows.empty:
        raise ValueError("Market model dataset contains no validation rows")
    validation_end = pd.to_datetime(validation_rows["observed_on"], errors="coerce").max()
    observed_dates = pd.to_datetime(daily_bars["observed_on"], errors="coerce")
    evaluation_bars = daily_bars.loc[observed_dates <= validation_end].copy()

    forecasts = {
        "ewma": build_ewma_validation_predictions(dataset, evaluation_bars),
        "ridge": build_ridge_validation_predictions(dataset),
        "xgboost": build_xgboost_validation_predictions(
            dataset,
            config=config,
            boosting_rounds=boosting_rounds,
        ),
    }
    keys = ["symbol", "observed_on"]
    comparison = forecasts["ewma"].loc[:, [*keys, TARGET_COLUMN]].copy()
    for model_name, predictions in forecasts.items():
        model_predictions = predictions.loc[:, [*keys, PREDICTED_VOLATILITY_COLUMN]].rename(
            columns={PREDICTED_VOLATILITY_COLUMN: FORECAST_COLUMNS[model_name]}
        )
        comparison = comparison.merge(
            model_predictions,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    if len(comparison) != len(validation_rows):
        raise ValueError("Validation forecasts do not align with all validation rows")
    return comparison.sort_values(keys).reset_index(drop=True)


def summarize_forecast_slice(rows: pd.DataFrame) -> dict[str, object]:
    """Summarize model errors and directional bias for one validation slice."""
    if rows.empty:
        raise ValueError("Cannot summarize an empty forecast slice")
    result: dict[str, object] = {
        "row_count": len(rows),
        "mean_actual_volatility": float(rows[TARGET_COLUMN].mean()),
    }
    for model_name, prediction_column in FORECAST_COLUMNS.items():
        metrics = calculate_volatility_forecast_metrics(
            rows[TARGET_COLUMN],
            rows[prediction_column],
        )
        result[model_name] = {
            **asdict(metrics),
            "bias": float((rows[prediction_column] - rows[TARGET_COLUMN]).mean()),
        }
    return result


def build_diagnostics_report(
    comparison: pd.DataFrame,
    *,
    dataset_metadata: dict[str, object],
    selection_version: str,
    selection_sha256: str,
    candidate_name: str,
    boosting_rounds: int,
) -> dict[str, object]:
    """Build descriptive year, symbol, and realized-volatility diagnostics."""
    rows = comparison.copy()
    rows["observed_on"] = pd.to_datetime(rows["observed_on"], errors="coerce")
    if rows["observed_on"].isna().any():
        raise ValueError("Forecast comparison contains invalid observation dates")
    rows["year"] = rows["observed_on"].dt.year
    rows["volatility_regime"] = pd.cut(
        rows[TARGET_COLUMN],
        bins=[-np.inf, 0.15, 0.30, np.inf],
        labels=["low_below_15pct", "moderate_15_to_30pct", "high_at_least_30pct"],
        right=False,
    )
    return {
        "schema_version": BOOSTING_DIAGNOSTICS_SCHEMA_VERSION,
        "evaluation_scope": "outer_validation_diagnostics",
        "parameter_changes_allowed": False,
        "test_split_evaluated": False,
        "dataset": {
            "version": dataset_metadata.get("dataset_version"),
            "sha256": dataset_metadata.get("sha256"),
        },
        "model_selection": {
            "version": normalize_artifact_version(selection_version),
            "sha256": selection_sha256,
            "candidate": candidate_name,
            "boosting_rounds": boosting_rounds,
        },
        "bias_definition": "mean_prediction_minus_actual",
        "overall": summarize_forecast_slice(rows),
        "by_year": {
            str(year): summarize_forecast_slice(group)
            for year, group in rows.groupby("year", observed=True)
        },
        "by_volatility_regime": {
            str(regime): summarize_forecast_slice(group)
            for regime, group in rows.groupby("volatility_regime", observed=True)
        },
        "by_symbol": {
            str(symbol): summarize_forecast_slice(group)
            for symbol, group in rows.groupby("symbol", observed=True)
        },
    }


def write_diagnostics_report(
    report: dict[str, object],
    diagnostics_version: str,
    *,
    output_directory: Path = DEFAULT_EVALUATION_DIRECTORY,
) -> Path:
    """Write an immutable versioned outer-validation diagnostics report."""
    version = normalize_artifact_version(diagnostics_version)
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"market_volatility_boosting_diagnostics_{version}.json"
    if destination.exists():
        raise FileExistsError(f"Boosting diagnostics report version already exists: {version}")
    temporary_destination = destination.with_suffix(".json.tmp")
    try:
        temporary_destination.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_destination.replace(destination)
    except Exception:
        temporary_destination.unlink(missing_ok=True)
        raise
    return destination


def run() -> None:
    parser = argparse.ArgumentParser(description="Diagnose selected volatility forecasts")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--selection-version", required=True)
    parser.add_argument("--diagnostics-version", required=True)
    parser.add_argument("--dataset-directory", type=Path, default=DEFAULT_DATASET_DIRECTORY)
    parser.add_argument("--snapshot-directory", type=Path, default=DEFAULT_SNAPSHOT_DIRECTORY)
    parser.add_argument(
        "--selection-directory",
        type=Path,
        default=DEFAULT_BOOSTING_SELECTION_DIRECTORY,
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_EVALUATION_DIRECTORY)
    args = parser.parse_args()

    dataset, dataset_metadata = load_model_dataset(args.dataset_version, args.dataset_directory)
    snapshot_version = dataset_metadata.get("source_snapshot_version")
    if not isinstance(snapshot_version, str):
        raise ValueError("Dataset metadata does not declare a source snapshot version")
    daily_bars, _ = load_market_snapshot(snapshot_version, args.snapshot_directory)
    selection_report, selection_sha256 = load_boosting_selection_report(
        args.selection_version,
        args.selection_directory,
    )
    selection_dataset = selection_report.get("dataset")
    if not isinstance(selection_dataset, dict) or (
        selection_dataset.get("version") != dataset_metadata.get("dataset_version")
        or selection_dataset.get("sha256") != dataset_metadata.get("sha256")
    ):
        raise ValueError("Boosting selection and model dataset provenance do not match")
    selected = selection_report["ranking"][0]
    config = XGBoostConfig(**selected["configuration"])
    boosting_rounds = selected["median_boosting_rounds"]
    comparison = build_forecast_comparison(
        dataset,
        daily_bars,
        config=config,
        boosting_rounds=boosting_rounds,
    )
    report = build_diagnostics_report(
        comparison,
        dataset_metadata=dataset_metadata,
        selection_version=args.selection_version,
        selection_sha256=selection_sha256,
        candidate_name=selected["candidate"],
        boosting_rounds=boosting_rounds,
    )
    destination = write_diagnostics_report(
        report,
        args.diagnostics_version,
        output_directory=args.output_directory,
    )
    print(f"Market volatility boosting diagnostics report: {destination}")


if __name__ == "__main__":
    run()
