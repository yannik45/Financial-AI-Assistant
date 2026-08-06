"""Derived dataset construction for market volatility modeling."""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from financial_ai.ml.artifact_integrity import (
    calculate_canonical_text_sha256,
    normalize_artifact_version,
)
from financial_ai.ml.market_forecast.features import FEATURE_COLUMNS, build_market_features
from financial_ai.ml.market_forecast.snapshot import (
    DEFAULT_OUTPUT_DIRECTORY as DEFAULT_SNAPSHOT_DIRECTORY,
)
from financial_ai.ml.market_forecast.snapshot import load_market_snapshot
from financial_ai.ml.market_forecast.splits import (
    DEFAULT_PURGE_TRADING_DAYS,
    DEFAULT_TEST_START,
    DEFAULT_VALIDATION_START,
    SPLIT_COLUMN,
    assign_chronological_splits,
)
from financial_ai.ml.market_forecast.targets import TARGET_COLUMN, build_forward_volatility_target

MODEL_DATASET_COLUMNS = (
    "symbol",
    "observed_on",
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
    SPLIT_COLUMN,
)
MODEL_DATASET_SCHEMA_VERSION = "market-volatility-dataset-v1"
DEFAULT_DATASET_DIRECTORY = Path("data/runtime/ml/market_forecast/datasets")


def build_model_dataset(
    daily_bars: pd.DataFrame,
    *,
    validation_start: date = DEFAULT_VALIDATION_START,
    test_start: date = DEFAULT_TEST_START,
    purge_trading_days: int = DEFAULT_PURGE_TRADING_DAYS,
) -> pd.DataFrame:
    """Build the feature, target, and split contract used by forecast models."""
    features = build_market_features(daily_bars)
    targets = build_forward_volatility_target(daily_bars)
    dataset = features.merge(
        targets.loc[:, ["symbol", "observed_on", TARGET_COLUMN]],
        on=["symbol", "observed_on"],
        how="inner",
        validate="one_to_one",
    )
    split_dataset = assign_chronological_splits(
        dataset,
        validation_start=validation_start,
        test_start=test_start,
        purge_trading_days=purge_trading_days,
    )
    return split_dataset.loc[:, list(MODEL_DATASET_COLUMNS)]


def build_model_dataset_metadata(
    dataset: pd.DataFrame,
    dataset_version: str,
    csv_path: Path,
    *,
    source_metadata: dict[str, object],
    validation_start: date = DEFAULT_VALIDATION_START,
    test_start: date = DEFAULT_TEST_START,
    purge_trading_days: int = DEFAULT_PURGE_TRADING_DAYS,
) -> dict[str, object]:
    """Describe the complete model dataset contract and its source snapshot."""
    split_summary: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        rows = dataset.loc[dataset[SPLIT_COLUMN] == split]
        split_summary[split] = {
            "row_count": len(rows),
            "date_from": rows["observed_on"].min().date().isoformat(),
            "date_to": rows["observed_on"].max().date().isoformat(),
        }

    dates = pd.to_datetime(dataset["observed_on"])
    return {
        "schema_version": MODEL_DATASET_SCHEMA_VERSION,
        "dataset_version": normalize_artifact_version(dataset_version),
        "source_snapshot_version": source_metadata.get("snapshot_version"),
        "source_snapshot_sha256": source_metadata.get("sha256"),
        "source_provider": source_metadata.get("provider"),
        "source_feed": source_metadata.get("feed"),
        "target_column": TARGET_COLUMN,
        "target_horizon_trading_days": 20,
        "target_annualized": True,
        "feature_columns": list(FEATURE_COLUMNS),
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "purge_trading_days": purge_trading_days,
        "symbols": sorted(dataset["symbol"].unique().tolist()),
        "date_from": dates.min().date().isoformat(),
        "date_to": dates.max().date().isoformat(),
        "row_count": len(dataset),
        "columns": list(MODEL_DATASET_COLUMNS),
        "splits": split_summary,
        "checksum_normalization": "utf-8-lf",
        "sha256": calculate_canonical_text_sha256(csv_path),
    }


def write_model_dataset(
    dataset: pd.DataFrame,
    dataset_version: str,
    *,
    source_metadata: dict[str, object],
    output_directory: Path = DEFAULT_DATASET_DIRECTORY,
    validation_start: date = DEFAULT_VALIDATION_START,
    test_start: date = DEFAULT_TEST_START,
    purge_trading_days: int = DEFAULT_PURGE_TRADING_DAYS,
) -> tuple[Path, Path]:
    """Write an immutable model dataset and provenance metadata."""
    version = normalize_artifact_version(dataset_version)
    validated = _validate_model_dataset(dataset)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"market_volatility_dataset_{version}.csv"
    metadata_path = output_directory / f"market_volatility_dataset_{version}.metadata.json"
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Market model dataset version already exists: {version}")

    temporary_csv = csv_path.with_suffix(".csv.tmp")
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    try:
        validated.to_csv(temporary_csv, index=False, lineterminator="\n")
        metadata = build_model_dataset_metadata(
            validated,
            version,
            temporary_csv,
            source_metadata=source_metadata,
            validation_start=validation_start,
            test_start=test_start,
            purge_trading_days=purge_trading_days,
        )
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_csv.replace(csv_path)
        temporary_metadata.replace(metadata_path)
    except Exception:
        temporary_csv.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        raise
    return csv_path, metadata_path


def load_model_dataset(
    dataset_version: str,
    input_directory: Path = DEFAULT_DATASET_DIRECTORY,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and verify a versioned model dataset."""
    version = normalize_artifact_version(dataset_version)
    csv_path = input_directory / f"market_volatility_dataset_{version}.csv"
    metadata_path = input_directory / f"market_volatility_dataset_{version}.metadata.json"
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Market model dataset not found: {version}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != MODEL_DATASET_SCHEMA_VERSION:
        raise ValueError("Market model dataset schema version is incompatible")
    if metadata.get("dataset_version") != version:
        raise ValueError("Market model dataset version does not match its metadata")
    if metadata.get("sha256") != calculate_canonical_text_sha256(csv_path):
        raise ValueError("Market model dataset checksum does not match its metadata")

    dataset = pd.read_csv(csv_path, keep_default_na=False)
    dataset["observed_on"] = pd.to_datetime(dataset["observed_on"], errors="coerce")
    return _validate_model_dataset(dataset), metadata


def _validate_model_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    result = dataset.copy(deep=True)
    if tuple(result.columns) != MODEL_DATASET_COLUMNS:
        raise ValueError("Market model dataset columns do not match the contract")
    result["symbol"] = result["symbol"].astype("string").str.strip().str.upper()
    result["observed_on"] = pd.to_datetime(result["observed_on"], errors="coerce")
    numeric_columns = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["symbol", "observed_on", SPLIT_COLUMN] + numeric_columns].isna().any().any():
        raise ValueError("Market model dataset contains missing or invalid values")
    if result["symbol"].eq("").any():
        raise ValueError("Market model dataset contains empty symbols")
    if set(result[SPLIT_COLUMN]) != {"train", "validation", "test"}:
        raise ValueError("Market model dataset must contain all expected splits")
    if result.duplicated(subset=["symbol", "observed_on"]).any():
        raise ValueError("Market model dataset contains duplicate symbol/date rows")
    return result.sort_values(["symbol", "observed_on"]).reset_index(drop=True)


def run() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable market model dataset")
    parser.add_argument("--snapshot-version", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--snapshot-directory", type=Path, default=DEFAULT_SNAPSHOT_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_DATASET_DIRECTORY)
    args = parser.parse_args()

    daily_bars, source_metadata = load_market_snapshot(
        args.snapshot_version,
        args.snapshot_directory,
    )
    dataset = build_model_dataset(daily_bars)
    csv_path, metadata_path = write_model_dataset(
        dataset,
        args.dataset_version,
        source_metadata=source_metadata,
        output_directory=args.output_directory,
    )
    print(f"Market model dataset: {csv_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    run()
