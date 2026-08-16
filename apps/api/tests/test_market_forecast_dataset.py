import numpy as np
import pandas as pd
import pytest
from financial_ai.ml.market_forecast.data.dataset import (
    MODEL_DATASET_COLUMNS,
    build_model_dataset,
    load_model_dataset,
    write_model_dataset,
)
from financial_ai.ml.market_forecast.data.features import FEATURE_COLUMNS
from financial_ai.ml.market_forecast.data.splits import SPLIT_COLUMN
from financial_ai.ml.market_forecast.data.targets import TARGET_COLUMN


def daily_bars() -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=140, freq="B")
    rows = []
    for offset, symbol in enumerate(("AAPL", "MSFT")):
        closes = 100 * np.exp(np.sin(np.arange(140) / 9 + offset) * 0.08)
        rows.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "observed_on": dates,
                    "open": closes * 0.995,
                    "high": closes * 1.01,
                    "low": closes * 0.99,
                    "close": closes,
                    "adjusted_close": closes,
                    "volume": np.linspace(1_000_000, 2_000_000, len(dates)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_model_dataset_contains_only_identifiers_features_target_and_split():
    source = daily_bars()
    dates = source["observed_on"].drop_duplicates().sort_values().reset_index(drop=True)

    result = build_model_dataset(
        source,
        validation_start=dates.iloc[80].date(),
        test_start=dates.iloc[110].date(),
        purge_trading_days=5,
    )

    assert tuple(result.columns) == MODEL_DATASET_COLUMNS
    assert not result.loc[:, list(FEATURE_COLUMNS) + [TARGET_COLUMN]].isna().any().any()
    assert result.groupby(SPLIT_COLUMN).size().to_dict() == {
        "test": 20,
        "train": 30,
        "validation": 50,
    }
    assert "adjusted_close" not in result.columns


def test_model_dataset_construction_does_not_modify_raw_bars():
    source = daily_bars()
    expected = source.copy(deep=True)
    dates = source["observed_on"].drop_duplicates().sort_values().reset_index(drop=True)

    build_model_dataset(
        source,
        validation_start=dates.iloc[80].date(),
        test_start=dates.iloc[110].date(),
        purge_trading_days=5,
    )

    pd.testing.assert_frame_equal(source, expected)


def test_model_dataset_artifact_preserves_provenance_and_checksum(tmp_path):
    source = daily_bars()
    dates = source["observed_on"].drop_duplicates().sort_values().reset_index(drop=True)
    validation_start = dates.iloc[80].date()
    test_start = dates.iloc[110].date()
    dataset = build_model_dataset(
        source,
        validation_start=validation_start,
        test_start=test_start,
        purge_trading_days=5,
    )
    source_metadata = {
        "snapshot_version": "source-v1",
        "sha256": "source-checksum",
        "provider": "alpaca",
        "feed": "sip",
    }

    csv_path, _ = write_model_dataset(
        dataset,
        "dataset-v1",
        source_metadata=source_metadata,
        output_directory=tmp_path,
        validation_start=validation_start,
        test_start=test_start,
        purge_trading_days=5,
    )
    loaded, metadata = load_model_dataset("dataset-v1", tmp_path)

    pd.testing.assert_frame_equal(loaded, dataset)
    assert metadata["source_snapshot_version"] == "source-v1"
    assert metadata["source_snapshot_sha256"] == "source-checksum"
    assert metadata["feature_columns"] == list(FEATURE_COLUMNS)
    assert metadata["validation_start"] == validation_start.isoformat()
    assert metadata["splits"]["test"]["row_count"] == 20

    with pytest.raises(FileExistsError, match="already exists"):
        write_model_dataset(
            dataset,
            "dataset-v1",
            source_metadata=source_metadata,
            output_directory=tmp_path,
        )

    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_model_dataset("dataset-v1", tmp_path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (FEATURE_COLUMNS[0], np.inf, "non-finite"),
        (TARGET_COLUMN, 0.0, "positive"),
    ],
)
def test_model_dataset_rejects_values_incompatible_with_training(tmp_path, column, value, message):
    source = daily_bars()
    dates = source["observed_on"].drop_duplicates().sort_values().reset_index(drop=True)
    dataset = build_model_dataset(
        source,
        validation_start=dates.iloc[80].date(),
        test_start=dates.iloc[110].date(),
        purge_trading_days=5,
    )
    dataset.loc[dataset.index[0], column] = value

    with pytest.raises(ValueError, match=message):
        write_model_dataset(
            dataset,
            "invalid-v1",
            source_metadata={"snapshot_version": "source-v1", "sha256": "checksum"},
            output_directory=tmp_path,
        )
