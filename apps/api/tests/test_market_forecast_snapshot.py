import json

import pandas as pd
import pytest
from financial_ai.ml.market_forecast.data.snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    build_snapshot_metadata,
    load_market_snapshot,
    write_market_snapshot,
)


def snapshot_bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "observed_on": "2024-01-02",
                "open": 186.0,
                "high": 188.0,
                "low": 183.5,
                "close": 185.64,
                "adjusted_close": 185.64,
                "volume": 82488700,
            },
            {
                "symbol": "MSFT",
                "observed_on": "2024-01-03",
                "open": 370.0,
                "high": 376.0,
                "low": 369.0,
                "close": 374.5,
                "adjusted_close": 374.5,
                "volume": 23000000,
            },
        ]
    )


def test_snapshot_metadata_describes_content_and_provenance(tmp_path):
    csv_path = tmp_path / "bars.csv"
    frame = snapshot_bars()
    frame.to_csv(csv_path, index=False, lineterminator="\n")

    metadata = build_snapshot_metadata(
        frame,
        "us-equities-v1",
        csv_path,
        provider="alpaca",
        feed="iex",
    )

    assert metadata == {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_version": "us-equities-v1",
        "provider": "alpaca",
        "feed": "iex",
        "frequency": "1day",
        "adjustment": "all",
        "date_from": "2024-01-02",
        "date_to": "2024-01-03",
        "symbols": ["AAPL", "MSFT"],
        "row_count": 2,
        "columns": [
            "symbol",
            "observed_on",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ],
        "sha256": metadata["sha256"],
        "checksum_normalization": "utf-8-lf",
    }
    assert len(metadata["sha256"]) == 64


def test_market_snapshot_is_immutable_and_checksum_verified(tmp_path):
    csv_path, metadata_path = write_market_snapshot(
        snapshot_bars(),
        "US-Equities-V1",
        output_directory=tmp_path,
    )

    loaded, metadata = load_market_snapshot("us-equities-v1", tmp_path)
    assert loaded["symbol"].tolist() == ["AAPL", "MSFT"]
    assert metadata["row_count"] == 2
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["sha256"]
    with pytest.raises(FileExistsError, match="already exists"):
        write_market_snapshot(snapshot_bars(), "us-equities-v1", output_directory=tmp_path)

    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_market_snapshot("us-equities-v1", tmp_path)
