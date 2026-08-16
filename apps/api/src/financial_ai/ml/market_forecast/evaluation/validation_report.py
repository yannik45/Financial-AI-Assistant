"""Reproducible validation reporting for market volatility forecasts."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from financial_ai.ml.artifact_integrity import normalize_artifact_version
from financial_ai.ml.market_forecast.data.dataset import (
    DEFAULT_DATASET_DIRECTORY,
    load_model_dataset,
)
from financial_ai.ml.market_forecast.data.features import FEATURE_COLUMNS
from financial_ai.ml.market_forecast.data.snapshot import (
    DEFAULT_OUTPUT_DIRECTORY as DEFAULT_SNAPSHOT_DIRECTORY,
)
from financial_ai.ml.market_forecast.data.snapshot import load_market_snapshot
from financial_ai.ml.market_forecast.data.splits import SPLIT_COLUMN
from financial_ai.ml.market_forecast.data.targets import TARGET_COLUMN
from financial_ai.ml.market_forecast.evaluation.evaluation import (
    calculate_volatility_forecast_metrics,
)
from financial_ai.ml.market_forecast.modeling.baselines import (
    PREDICTED_VOLATILITY_COLUMN,
    ConstantStrategy,
    build_constant_validation_predictions,
    evaluate_ewma_validation,
    evaluate_persistence_validation,
)
from financial_ai.ml.market_forecast.modeling.ewma import (
    DEFAULT_EWMA_DECAY,
    DEFAULT_EWMA_MIN_OBSERVATIONS,
    EWMA_VOLATILITY_COLUMN,
)
from financial_ai.ml.market_forecast.modeling.ridge import (
    DEFAULT_RIDGE_ALPHA,
    evaluate_ridge_ewma_validation,
    evaluate_ridge_validation,
)

VALIDATION_REPORT_SCHEMA_VERSION = "market-volatility-validation-report-v1"
DEFAULT_EVALUATION_DIRECTORY = Path("data/runtime/ml/market_forecast/evaluations")


def build_validation_report(
    dataset: pd.DataFrame,
    daily_bars: pd.DataFrame,
    *,
    dataset_metadata: dict[str, object],
    snapshot_metadata: dict[str, object],
) -> dict[str, object]:
    """Evaluate declared references and Ridge variants without using the test split."""
    _validate_provenance(dataset_metadata, snapshot_metadata)
    validation_rows = dataset.loc[dataset[SPLIT_COLUMN] == "validation"]
    if validation_rows.empty:
        raise ValueError("Market model dataset contains no validation rows")

    validation_end = pd.to_datetime(validation_rows["observed_on"]).max()
    observed_dates = pd.to_datetime(daily_bars["observed_on"], errors="coerce")
    evaluation_bars = daily_bars.loc[observed_dates <= validation_end].copy()

    evaluations = {
        "constant_mean": _evaluate_constant(dataset, "mean"),
        "constant_median": _evaluate_constant(dataset, "median"),
        "persistence_20d": asdict(evaluate_persistence_validation(dataset)),
        "ewma": asdict(evaluate_ewma_validation(dataset, evaluation_bars)),
        "ridge": asdict(evaluate_ridge_validation(dataset)),
        "ridge_with_ewma": asdict(evaluate_ridge_ewma_validation(dataset, evaluation_bars)),
    }
    return {
        "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "evaluation_scope": "validation",
        "test_split_evaluated": False,
        "selection_metric": "mae",
        "dataset": {
            "version": dataset_metadata.get("dataset_version"),
            "sha256": dataset_metadata.get("sha256"),
            "validation": dataset_metadata.get("splits", {}).get("validation"),
        },
        "source_snapshot": {
            "version": snapshot_metadata.get("snapshot_version"),
            "sha256": snapshot_metadata.get("sha256"),
            "provider": snapshot_metadata.get("provider"),
            "feed": snapshot_metadata.get("feed"),
        },
        "target": {
            "column": TARGET_COLUMN,
            "horizon_trading_days": 20,
            "annualized": True,
        },
        "metrics": {
            "mae": "annualized volatility",
            "rmse": "annualized volatility",
            "qlike": "squared-volatility ratio loss",
        },
        "configuration": {
            "ridge_alpha": DEFAULT_RIDGE_ALPHA,
            "ridge_target_transform": "log-exp",
            "ridge_features": list(FEATURE_COLUMNS),
            "ridge_ewma_features": [*FEATURE_COLUMNS, EWMA_VOLATILITY_COLUMN],
            "ewma_decay": DEFAULT_EWMA_DECAY,
            "ewma_min_observations": DEFAULT_EWMA_MIN_OBSERVATIONS,
        },
        "evaluations": evaluations,
    }


def _evaluate_constant(dataset: pd.DataFrame, strategy: ConstantStrategy) -> dict[str, float]:
    predictions = build_constant_validation_predictions(dataset, strategy)
    metrics = calculate_volatility_forecast_metrics(
        predictions[TARGET_COLUMN],
        predictions[PREDICTED_VOLATILITY_COLUMN],
    )
    return asdict(metrics)


def _validate_provenance(
    dataset_metadata: dict[str, object],
    snapshot_metadata: dict[str, object],
) -> None:
    if dataset_metadata.get("source_snapshot_version") != snapshot_metadata.get("snapshot_version"):
        raise ValueError("Dataset and snapshot versions do not match")
    if dataset_metadata.get("source_snapshot_sha256") != snapshot_metadata.get("sha256"):
        raise ValueError("Dataset and snapshot checksums do not match")


def write_validation_report(
    report: dict[str, object],
    evaluation_version: str,
    *,
    output_directory: Path = DEFAULT_EVALUATION_DIRECTORY,
) -> Path:
    """Write an immutable versioned validation report."""
    version = normalize_artifact_version(evaluation_version)
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"market_volatility_validation_{version}.json"
    if destination.exists():
        raise FileExistsError(f"Market validation report version already exists: {version}")
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
    parser = argparse.ArgumentParser(description="Evaluate market volatility validation forecasts")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--evaluation-version", required=True)
    parser.add_argument("--dataset-directory", type=Path, default=DEFAULT_DATASET_DIRECTORY)
    parser.add_argument("--snapshot-directory", type=Path, default=DEFAULT_SNAPSHOT_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_EVALUATION_DIRECTORY)
    args = parser.parse_args()

    dataset, dataset_metadata = load_model_dataset(
        args.dataset_version,
        args.dataset_directory,
    )
    snapshot_version = dataset_metadata.get("source_snapshot_version")
    if not isinstance(snapshot_version, str):
        raise ValueError("Dataset metadata does not declare a source snapshot version")
    daily_bars, snapshot_metadata = load_market_snapshot(
        snapshot_version,
        args.snapshot_directory,
    )
    report = build_validation_report(
        dataset,
        daily_bars,
        dataset_metadata=dataset_metadata,
        snapshot_metadata=snapshot_metadata,
    )
    destination = write_validation_report(
        report,
        args.evaluation_version,
        output_directory=args.output_directory,
    )
    print(f"Market volatility validation report: {destination}")


if __name__ == "__main__":
    run()
