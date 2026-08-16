import json

import numpy as np
import pandas as pd
import pytest
from financial_ai.ml.market_forecast.data.dataset import build_model_dataset
from financial_ai.ml.market_forecast.evaluation.validation_report import (
    build_validation_report,
    write_validation_report,
)


def daily_bars() -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=180, freq="B")
    frames = []
    for offset, symbol in enumerate(("AAPL", "MSFT")):
        log_prices = 4.5 + np.sin(np.arange(180) / 8 + offset) * 0.08
        closes = np.exp(log_prices)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "observed_on": dates,
                    "open": closes,
                    "high": closes * 1.01,
                    "low": closes * 0.99,
                    "close": closes,
                    "adjusted_close": closes,
                    "volume": np.linspace(1_000_000, 2_000_000, len(dates)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def report_inputs():
    bars = daily_bars()
    dates = bars["observed_on"].drop_duplicates().sort_values().reset_index(drop=True)
    dataset = build_model_dataset(
        bars,
        validation_start=dates.iloc[110].date(),
        test_start=dates.iloc[145].date(),
        purge_trading_days=20,
    )
    validation = dataset.loc[dataset["split"] == "validation"]
    dataset_metadata = {
        "dataset_version": "dataset-v1",
        "sha256": "dataset-checksum",
        "source_snapshot_version": "snapshot-v1",
        "source_snapshot_sha256": "snapshot-checksum",
        "splits": {
            "validation": {
                "row_count": len(validation),
                "date_from": validation["observed_on"].min().date().isoformat(),
                "date_to": validation["observed_on"].max().date().isoformat(),
            }
        },
    }
    snapshot_metadata = {
        "snapshot_version": "snapshot-v1",
        "sha256": "snapshot-checksum",
        "provider": "test",
        "feed": "test",
    }
    return dataset, bars, dataset_metadata, snapshot_metadata


def test_validation_report_records_provenance_metrics_and_test_exclusion():
    dataset, bars, dataset_metadata, snapshot_metadata = report_inputs()

    report = build_validation_report(
        dataset,
        bars,
        dataset_metadata=dataset_metadata,
        snapshot_metadata=snapshot_metadata,
    )

    assert report["evaluation_scope"] == "validation"
    assert report["test_split_evaluated"] is False
    assert report["dataset"]["sha256"] == "dataset-checksum"
    assert set(report["evaluations"]) == {
        "constant_mean",
        "constant_median",
        "persistence_20d",
        "ewma",
        "ridge",
        "ridge_with_ewma",
    }


def test_validation_report_is_immutable(tmp_path):
    report = {"schema_version": "test"}

    destination = write_validation_report(report, "evaluation-v1", output_directory=tmp_path)

    assert json.loads(destination.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="already exists"):
        write_validation_report(report, "evaluation-v1", output_directory=tmp_path)


def test_validation_report_rejects_snapshot_provenance_mismatch():
    dataset, bars, dataset_metadata, snapshot_metadata = report_inputs()
    snapshot_metadata["sha256"] = "different-checksum"

    with pytest.raises(ValueError, match="checksums"):
        build_validation_report(
            dataset,
            bars,
            dataset_metadata=dataset_metadata,
            snapshot_metadata=snapshot_metadata,
        )
