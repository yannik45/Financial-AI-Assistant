import argparse
import json
from pathlib import Path

import pandas as pd

from financial_ai.ml.artifact_integrity import (
    calculate_canonical_text_sha256,
    normalize_artifact_version,
)
from financial_ai.ml.market_forecast.data.daily_bars import DAILY_BAR_COLUMNS, validate_daily_bars

SNAPSHOT_SCHEMA_VERSION = "market-daily-bars-v1"
DEFAULT_OUTPUT_DIRECTORY = Path("data/runtime/ml/market_forecast/snapshots")


def build_snapshot_metadata(
    frame: pd.DataFrame,
    snapshot_version: str,
    csv_path: Path,
    *,
    provider: str,
    feed: str,
) -> dict[str, object]:
    """Describe the exact observations and provenance stored in a market snapshot."""

    meta_dict = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "frequency": "1day",
        "adjustment": "all",
        "checksum_normalization": "utf-8-lf",
        "snapshot_version": normalize_artifact_version(snapshot_version),
        "provider": provider,
        "feed": feed,
    }

    symbols = frame["symbol"].drop_duplicates().sort_values().tolist()

    dates = pd.to_datetime(frame["observed_on"])

    date_from = dates.min().date().isoformat()
    date_to = dates.max().date().isoformat()

    meta_dict["symbols"] = symbols
    meta_dict["date_from"] = date_from
    meta_dict["date_to"] = date_to
    meta_dict["row_count"] = len(frame)
    meta_dict["columns"] = list(DAILY_BAR_COLUMNS)
    meta_dict["sha256"] = calculate_canonical_text_sha256(csv_path)

    return meta_dict


def write_market_snapshot(
    frame: pd.DataFrame,
    snapshot_version: str,
    *,
    provider: str = "alpaca",
    feed: str = "iex",
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[Path, Path]:
    version = normalize_artifact_version(snapshot_version)
    validated = validate_daily_bars(frame)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"market_daily_bars_{version}.csv"
    metadata_path = output_directory / f"market_daily_bars_{version}.metadata.json"
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Market snapshot version already exists: {version}. "
            "Choose a new version instead of overwriting training evidence."
        )

    temporary_csv = csv_path.with_suffix(".csv.tmp")
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    try:
        validated.to_csv(temporary_csv, index=False, lineterminator="\n")
        metadata = build_snapshot_metadata(
            validated,
            version,
            temporary_csv,
            provider=provider,
            feed=feed,
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


def load_market_snapshot(
    snapshot_version: str,
    input_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[pd.DataFrame, dict[str, object]]:
    version = normalize_artifact_version(snapshot_version)
    csv_path = input_directory / f"market_daily_bars_{version}.csv"
    metadata_path = input_directory / f"market_daily_bars_{version}.metadata.json"
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Market snapshot not found: {version}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("Market snapshot schema version is incompatible")
    if metadata.get("snapshot_version") != version:
        raise ValueError("Market snapshot version does not match its metadata")
    if metadata.get("sha256") != calculate_canonical_text_sha256(csv_path):
        raise ValueError("Market snapshot checksum does not match its metadata")

    frame = pd.read_csv(csv_path, keep_default_na=False)
    validated = validate_daily_bars(frame)
    if list(validated.columns) != list(DAILY_BAR_COLUMNS):
        raise ValueError("Market snapshot columns do not match the export schema")
    return validated, metadata


def run() -> None:
    parser = argparse.ArgumentParser(description="Freeze validated daily market observations")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--provider", default="alpaca")
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    source = pd.read_csv(args.input_csv, keep_default_na=False)
    csv_path, metadata_path = write_market_snapshot(
        source,
        args.version,
        provider=args.provider,
        feed=args.feed,
        output_directory=args.output_directory,
    )
    print(f"Market snapshot: {csv_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    run()
