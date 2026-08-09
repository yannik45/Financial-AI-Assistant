"""Reproducible inner-validation reporting for XGBoost model selection."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from financial_ai.ml.artifact_integrity import (
    calculate_canonical_text_sha256,
    normalize_artifact_version,
)
from financial_ai.ml.market_forecast.boosting import (
    BoostingCandidateEvaluation,
    compare_xgboost_candidates,
)
from financial_ai.ml.market_forecast.dataset import (
    DEFAULT_DATASET_DIRECTORY,
    load_model_dataset,
)
from financial_ai.ml.market_forecast.features import FEATURE_COLUMNS
from financial_ai.ml.market_forecast.model_selection import (
    DEFAULT_INNER_VALIDATION_YEARS,
    build_expanding_training_folds,
)
from financial_ai.ml.market_forecast.splits import DEFAULT_PURGE_TRADING_DAYS
from financial_ai.ml.market_forecast.targets import TARGET_COLUMN

BOOSTING_SELECTION_REPORT_SCHEMA_VERSION = "market-volatility-boosting-selection-v1"
DEFAULT_BOOSTING_SELECTION_DIRECTORY = Path("data/runtime/ml/market_forecast/model_selection")


def build_boosting_selection_report(
    dataset: pd.DataFrame,
    *,
    dataset_metadata: dict[str, object],
) -> dict[str, object]:
    """Evaluate declared candidates using only purged inner training folds."""
    folds = build_expanding_training_folds(dataset)
    evaluations = compare_xgboost_candidates(folds)
    return {
        "schema_version": BOOSTING_SELECTION_REPORT_SCHEMA_VERSION,
        "evaluation_scope": "inner_cross_validation",
        "outer_validation_evaluated": False,
        "test_split_evaluated": False,
        "selection_metric": "mean_mae",
        "dataset": {
            "version": dataset_metadata.get("dataset_version"),
            "sha256": dataset_metadata.get("sha256"),
            "outer_train": dataset_metadata.get("splits", {}).get("train"),
        },
        "target": {
            "column": TARGET_COLUMN,
            "horizon_trading_days": 20,
            "annualized": True,
        },
        "features": list(FEATURE_COLUMNS),
        "fold_strategy": {
            "type": "expanding_window",
            "validation_years": list(DEFAULT_INNER_VALIDATION_YEARS),
            "purge_trading_days": DEFAULT_PURGE_TRADING_DAYS,
        },
        "ranking": [
            _serialize_candidate_evaluation(rank, evaluation)
            for rank, evaluation in enumerate(evaluations, start=1)
        ],
    }


def _serialize_candidate_evaluation(
    rank: int,
    evaluation: BoostingCandidateEvaluation,
) -> dict[str, object]:
    """Convert a ranked candidate evaluation to its report representation."""
    return {
        "rank": rank,
        "candidate": evaluation.candidate.name,
        "configuration": asdict(evaluation.candidate.config),
        "mean_metrics": asdict(evaluation.mean_metrics),
        "median_boosting_rounds": evaluation.median_boosting_rounds,
        "folds": [asdict(fold) for fold in evaluation.folds],
    }


def write_boosting_selection_report(
    report: dict[str, object],
    selection_version: str,
    *,
    output_directory: Path = DEFAULT_BOOSTING_SELECTION_DIRECTORY,
) -> Path:
    """Write an immutable versioned model-selection report."""
    version = normalize_artifact_version(selection_version)
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"market_volatility_boosting_selection_{version}.json"
    if destination.exists():
        raise FileExistsError(f"Boosting selection report version already exists: {version}")
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


def load_boosting_selection_report(
    selection_version: str,
    input_directory: Path = DEFAULT_BOOSTING_SELECTION_DIRECTORY,
) -> tuple[dict[str, object], str]:
    """Load a compatible model-selection report and its canonical checksum."""
    version = normalize_artifact_version(selection_version)
    source = input_directory / f"market_volatility_boosting_selection_{version}.json"
    if not source.is_file():
        raise FileNotFoundError(f"Boosting selection report not found: {version}")
    report = json.loads(source.read_text(encoding="utf-8"))
    if report.get("schema_version") != BOOSTING_SELECTION_REPORT_SCHEMA_VERSION:
        raise ValueError("Boosting selection report schema version is incompatible")
    if report.get("evaluation_scope") != "inner_cross_validation":
        raise ValueError("Boosting selection report has an incompatible evaluation scope")
    if report.get("outer_validation_evaluated") is not False:
        raise ValueError("Boosting selection report must precede outer validation")
    if report.get("test_split_evaluated") is not False:
        raise ValueError("Boosting selection report must not evaluate the test split")
    ranking = report.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("Boosting selection report contains no ranked candidates")
    return report, calculate_canonical_text_sha256(source)


def run() -> None:
    parser = argparse.ArgumentParser(description="Select a market volatility XGBoost candidate")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--selection-version", required=True)
    parser.add_argument("--dataset-directory", type=Path, default=DEFAULT_DATASET_DIRECTORY)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_BOOSTING_SELECTION_DIRECTORY,
    )
    args = parser.parse_args()

    dataset, dataset_metadata = load_model_dataset(
        args.dataset_version,
        args.dataset_directory,
    )
    report = build_boosting_selection_report(
        dataset,
        dataset_metadata=dataset_metadata,
    )
    destination = write_boosting_selection_report(
        report,
        args.selection_version,
        output_directory=args.output_directory,
    )
    print(f"Market volatility boosting selection report: {destination}")


if __name__ == "__main__":
    run()
