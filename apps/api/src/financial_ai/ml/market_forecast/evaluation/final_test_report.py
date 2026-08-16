"""Final chronological test reporting for frozen volatility forecasts."""

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
from financial_ai.ml.market_forecast.data.snapshot import (
    DEFAULT_OUTPUT_DIRECTORY as DEFAULT_SNAPSHOT_DIRECTORY,
)
from financial_ai.ml.market_forecast.data.snapshot import load_market_snapshot
from financial_ai.ml.market_forecast.data.splits import SPLIT_COLUMN
from financial_ai.ml.market_forecast.data.targets import TARGET_COLUMN
from financial_ai.ml.market_forecast.evaluation.boosting_validation_report import (
    load_boosting_validation_report,
)
from financial_ai.ml.market_forecast.evaluation.evaluation import (
    calculate_volatility_forecast_metrics,
)
from financial_ai.ml.market_forecast.evaluation.validation_report import (
    DEFAULT_EVALUATION_DIRECTORY,
)
from financial_ai.ml.market_forecast.modeling.baselines import (
    PREDICTED_VOLATILITY_COLUMN,
    build_ewma_validation_predictions,
)
from financial_ai.ml.market_forecast.modeling.boosting import (
    XGBoostConfig,
    build_xgboost_validation_predictions,
)
from financial_ai.ml.market_forecast.modeling.ridge import build_ridge_validation_predictions

FINAL_TEST_REPORT_SCHEMA_VERSION = "market-volatility-final-test-v1"


def prepare_final_test_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Map development rows to fitting data and final-test rows to evaluation data."""
    required_splits = {"train", "validation", "test"}
    if SPLIT_COLUMN not in dataset.columns or set(dataset[SPLIT_COLUMN]) != required_splits:
        raise ValueError("Final test requires train, validation, and test splits")
    result = dataset.copy(deep=True)
    result.loc[result[SPLIT_COLUMN] == "validation", SPLIT_COLUMN] = "train"
    result.loc[result[SPLIT_COLUMN] == "test", SPLIT_COLUMN] = "validation"
    return result


def summarize_test_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate final-test metrics and signed forecast bias."""
    metrics = calculate_volatility_forecast_metrics(
        predictions[TARGET_COLUMN],
        predictions[PREDICTED_VOLATILITY_COLUMN],
    )
    return {
        **asdict(metrics),
        "bias": float(
            (predictions[PREDICTED_VOLATILITY_COLUMN] - predictions[TARGET_COLUMN]).mean()
        ),
    }


def build_final_test_report(
    dataset: pd.DataFrame,
    daily_bars: pd.DataFrame,
    *,
    dataset_metadata: dict[str, object],
    snapshot_metadata: dict[str, object],
    validation_report: dict[str, object],
    validation_version: str,
    validation_sha256: str,
) -> dict[str, object]:
    """Refit frozen models on development data and evaluate the final test once."""
    _validate_provenance(dataset_metadata, snapshot_metadata, validation_report)
    selection = validation_report.get("model_selection")
    if not isinstance(selection, dict):
        raise ValueError("Boosting validation report has no model selection")
    configuration = selection.get("configuration")
    boosting_rounds = selection.get("boosting_rounds")
    if not isinstance(configuration, dict):
        raise ValueError("Boosting validation report has no selected configuration")
    if not isinstance(boosting_rounds, int) or boosting_rounds < 1:
        raise ValueError("Boosting validation report has invalid boosting rounds")
    try:
        config = XGBoostConfig(**configuration)
    except TypeError as error:
        raise ValueError("Selected boosting configuration is incompatible") from error

    final_dataset = prepare_final_test_dataset(dataset)
    test_rows = final_dataset.loc[final_dataset[SPLIT_COLUMN] == "validation"]
    test_end = pd.to_datetime(test_rows["observed_on"], errors="coerce").max()
    if pd.isna(test_end):
        raise ValueError("Final test rows contain invalid observation dates")
    observed_dates = pd.to_datetime(daily_bars["observed_on"], errors="coerce")
    evaluation_bars = daily_bars.loc[observed_dates <= test_end].copy()

    predictions = {
        "ewma": build_ewma_validation_predictions(final_dataset, evaluation_bars),
        "ridge": build_ridge_validation_predictions(final_dataset),
        "xgboost": build_xgboost_validation_predictions(
            final_dataset,
            config=config,
            boosting_rounds=boosting_rounds,
        ),
    }
    by_year = {}
    for year in sorted(pd.to_datetime(test_rows["observed_on"]).dt.year.unique()):
        by_year[str(year)] = {
            model_name: summarize_test_predictions(
                model_predictions.loc[
                    pd.to_datetime(model_predictions["observed_on"]).dt.year == year
                ]
            )
            for model_name, model_predictions in predictions.items()
        }

    return {
        "schema_version": FINAL_TEST_REPORT_SCHEMA_VERSION,
        "evaluation_scope": "final_test",
        "model_selection_complete": True,
        "further_tuning_allowed": False,
        "dataset": {
            "version": dataset_metadata.get("dataset_version"),
            "sha256": dataset_metadata.get("sha256"),
            "fitting_rows": len(final_dataset.loc[final_dataset[SPLIT_COLUMN] == "train"]),
            "test": dataset_metadata.get("splits", {}).get("test"),
        },
        "source_snapshot": {
            "version": snapshot_metadata.get("snapshot_version"),
            "sha256": snapshot_metadata.get("sha256"),
            "provider": snapshot_metadata.get("provider"),
            "feed": snapshot_metadata.get("feed"),
        },
        "frozen_validation": {
            "version": normalize_artifact_version(validation_version),
            "sha256": validation_sha256,
            "candidate": selection.get("candidate"),
            "configuration": configuration,
            "boosting_rounds": boosting_rounds,
        },
        "target": {
            "column": TARGET_COLUMN,
            "horizon_trading_days": 20,
            "annualized": True,
        },
        "evaluations": {
            model_name: summarize_test_predictions(model_predictions)
            for model_name, model_predictions in predictions.items()
        },
        "by_year": by_year,
    }


def _validate_provenance(
    dataset_metadata: dict[str, object],
    snapshot_metadata: dict[str, object],
    validation_report: dict[str, object],
) -> None:
    validation_dataset = validation_report.get("dataset")
    if not isinstance(validation_dataset, dict):
        raise ValueError("Boosting validation report has no dataset provenance")
    if validation_dataset.get("version") != dataset_metadata.get("dataset_version"):
        raise ValueError("Validation report and model dataset versions do not match")
    if validation_dataset.get("sha256") != dataset_metadata.get("sha256"):
        raise ValueError("Validation report and model dataset checksums do not match")
    if dataset_metadata.get("source_snapshot_version") != snapshot_metadata.get("snapshot_version"):
        raise ValueError("Dataset and snapshot versions do not match")
    if dataset_metadata.get("source_snapshot_sha256") != snapshot_metadata.get("sha256"):
        raise ValueError("Dataset and snapshot checksums do not match")


def write_final_test_report(
    report: dict[str, object],
    test_version: str,
    *,
    output_directory: Path = DEFAULT_EVALUATION_DIRECTORY,
) -> Path:
    """Write an immutable versioned final-test report."""
    version = normalize_artifact_version(test_version)
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"market_volatility_final_test_{version}.json"
    if destination.exists():
        raise FileExistsError(f"Market final-test report version already exists: {version}")
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
    parser = argparse.ArgumentParser(description="Evaluate frozen volatility models on test")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--validation-version", required=True)
    parser.add_argument("--test-version", required=True)
    parser.add_argument("--dataset-directory", type=Path, default=DEFAULT_DATASET_DIRECTORY)
    parser.add_argument("--snapshot-directory", type=Path, default=DEFAULT_SNAPSHOT_DIRECTORY)
    parser.add_argument("--evaluation-directory", type=Path, default=DEFAULT_EVALUATION_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_EVALUATION_DIRECTORY)
    args = parser.parse_args()

    dataset, dataset_metadata = load_model_dataset(args.dataset_version, args.dataset_directory)
    snapshot_version = dataset_metadata.get("source_snapshot_version")
    if not isinstance(snapshot_version, str):
        raise ValueError("Dataset metadata does not declare a source snapshot version")
    daily_bars, snapshot_metadata = load_market_snapshot(
        snapshot_version,
        args.snapshot_directory,
    )
    validation_report, validation_sha256 = load_boosting_validation_report(
        args.validation_version,
        args.evaluation_directory,
    )
    report = build_final_test_report(
        dataset,
        daily_bars,
        dataset_metadata=dataset_metadata,
        snapshot_metadata=snapshot_metadata,
        validation_report=validation_report,
        validation_version=args.validation_version,
        validation_sha256=validation_sha256,
    )
    destination = write_final_test_report(
        report,
        args.test_version,
        output_directory=args.output_directory,
    )
    print(f"Market volatility final-test report: {destination}")


if __name__ == "__main__":
    run()
