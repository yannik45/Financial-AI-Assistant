"""Outer-validation reporting for the selected market volatility model."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from financial_ai.ml.artifact_integrity import (
    calculate_canonical_text_sha256,
    normalize_artifact_version,
)
from financial_ai.ml.market_forecast.baselines import evaluate_ewma_validation
from financial_ai.ml.market_forecast.boosting import (
    XGBoostConfig,
    evaluate_xgboost_validation,
)
from financial_ai.ml.market_forecast.boosting_report import (
    DEFAULT_BOOSTING_SELECTION_DIRECTORY,
    load_boosting_selection_report,
)
from financial_ai.ml.market_forecast.dataset import (
    DEFAULT_DATASET_DIRECTORY,
    load_model_dataset,
)
from financial_ai.ml.market_forecast.ridge import evaluate_ridge_validation
from financial_ai.ml.market_forecast.snapshot import (
    DEFAULT_OUTPUT_DIRECTORY as DEFAULT_SNAPSHOT_DIRECTORY,
)
from financial_ai.ml.market_forecast.snapshot import load_market_snapshot
from financial_ai.ml.market_forecast.splits import SPLIT_COLUMN
from financial_ai.ml.market_forecast.targets import TARGET_COLUMN
from financial_ai.ml.market_forecast.validation_report import DEFAULT_EVALUATION_DIRECTORY

BOOSTING_VALIDATION_REPORT_SCHEMA_VERSION = "market-volatility-boosting-validation-v1"


def build_boosting_validation_report(
    dataset: pd.DataFrame,
    daily_bars: pd.DataFrame,
    *,
    dataset_metadata: dict[str, object],
    snapshot_metadata: dict[str, object],
    selection_report: dict[str, object],
    selection_version: str,
    selection_sha256: str,
) -> dict[str, object]:
    """Evaluate the preselected candidate once on outer validation."""
    _validate_provenance(dataset_metadata, snapshot_metadata, selection_report)
    selected = selection_report["ranking"][0]
    if not isinstance(selected, dict) or selected.get("rank") != 1:
        raise ValueError("Boosting selection report has no rank-one candidate")
    configuration = selected.get("configuration")
    boosting_rounds = selected.get("median_boosting_rounds")
    if not isinstance(configuration, dict):
        raise ValueError("Selected boosting candidate has no configuration")
    if not isinstance(boosting_rounds, int) or boosting_rounds < 1:
        raise ValueError("Selected boosting candidate has invalid boosting rounds")
    try:
        config = XGBoostConfig(**configuration)
    except TypeError as error:
        raise ValueError("Selected boosting configuration is incompatible") from error

    validation_rows = dataset.loc[dataset[SPLIT_COLUMN] == "validation"]
    if validation_rows.empty:
        raise ValueError("Market model dataset contains no validation rows")
    validation_end = pd.to_datetime(validation_rows["observed_on"], errors="coerce").max()
    if pd.isna(validation_end):
        raise ValueError("Market model validation rows contain invalid observation dates")
    observed_dates = pd.to_datetime(daily_bars["observed_on"], errors="coerce")
    evaluation_bars = daily_bars.loc[observed_dates <= validation_end].copy()

    evaluations = {
        "ewma": asdict(evaluate_ewma_validation(dataset, evaluation_bars)),
        "ridge": asdict(evaluate_ridge_validation(dataset)),
        "xgboost": asdict(
            evaluate_xgboost_validation(
                dataset,
                config=config,
                boosting_rounds=boosting_rounds,
            )
        ),
    }
    return {
        "schema_version": BOOSTING_VALIDATION_REPORT_SCHEMA_VERSION,
        "evaluation_scope": "outer_validation",
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
        "model_selection": {
            "version": normalize_artifact_version(selection_version),
            "sha256": selection_sha256,
            "candidate": selected.get("candidate"),
            "configuration": configuration,
            "boosting_rounds": boosting_rounds,
        },
        "target": {
            "column": TARGET_COLUMN,
            "horizon_trading_days": 20,
            "annualized": True,
        },
        "evaluations": evaluations,
    }


def _validate_provenance(
    dataset_metadata: dict[str, object],
    snapshot_metadata: dict[str, object],
    selection_report: dict[str, object],
) -> None:
    selection_dataset = selection_report.get("dataset")
    if not isinstance(selection_dataset, dict):
        raise ValueError("Boosting selection report has no dataset provenance")
    if selection_dataset.get("version") != dataset_metadata.get("dataset_version"):
        raise ValueError("Boosting selection and model dataset versions do not match")
    if selection_dataset.get("sha256") != dataset_metadata.get("sha256"):
        raise ValueError("Boosting selection and model dataset checksums do not match")
    if dataset_metadata.get("source_snapshot_version") != snapshot_metadata.get("snapshot_version"):
        raise ValueError("Dataset and snapshot versions do not match")
    if dataset_metadata.get("source_snapshot_sha256") != snapshot_metadata.get("sha256"):
        raise ValueError("Dataset and snapshot checksums do not match")


def write_boosting_validation_report(
    report: dict[str, object],
    evaluation_version: str,
    *,
    output_directory: Path = DEFAULT_EVALUATION_DIRECTORY,
) -> Path:
    """Write an immutable versioned outer-validation report."""
    version = normalize_artifact_version(evaluation_version)
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"market_volatility_boosting_validation_{version}.json"
    if destination.exists():
        raise FileExistsError(f"Boosting validation report version already exists: {version}")
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


def load_boosting_validation_report(
    evaluation_version: str,
    input_directory: Path = DEFAULT_EVALUATION_DIRECTORY,
) -> tuple[dict[str, object], str]:
    """Load a compatible outer-validation report and its canonical checksum."""
    version = normalize_artifact_version(evaluation_version)
    source = input_directory / f"market_volatility_boosting_validation_{version}.json"
    if not source.is_file():
        raise FileNotFoundError(f"Boosting validation report not found: {version}")
    report = json.loads(source.read_text(encoding="utf-8"))
    if report.get("schema_version") != BOOSTING_VALIDATION_REPORT_SCHEMA_VERSION:
        raise ValueError("Boosting validation report schema version is incompatible")
    if report.get("evaluation_scope") != "outer_validation":
        raise ValueError("Boosting validation report has an incompatible evaluation scope")
    if report.get("test_split_evaluated") is not False:
        raise ValueError("Boosting validation report must precede final test evaluation")
    return report, calculate_canonical_text_sha256(source)


def run() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the selected boosting candidate")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--selection-version", required=True)
    parser.add_argument("--evaluation-version", required=True)
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
    daily_bars, snapshot_metadata = load_market_snapshot(
        snapshot_version,
        args.snapshot_directory,
    )
    selection_report, selection_sha256 = load_boosting_selection_report(
        args.selection_version,
        args.selection_directory,
    )
    report = build_boosting_validation_report(
        dataset,
        daily_bars,
        dataset_metadata=dataset_metadata,
        snapshot_metadata=snapshot_metadata,
        selection_report=selection_report,
        selection_version=args.selection_version,
        selection_sha256=selection_sha256,
    )
    destination = write_boosting_validation_report(
        report,
        args.evaluation_version,
        output_directory=args.output_directory,
    )
    print(f"Market volatility boosting validation report: {destination}")


if __name__ == "__main__":
    run()
