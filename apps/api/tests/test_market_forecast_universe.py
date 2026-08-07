import json

import pytest
from financial_ai.ml.market_forecast.universe import (
    DEFAULT_UNIVERSE_PATH,
    MarketUniverseError,
    load_market_universe,
)


def test_default_market_universe_is_versioned_unique_and_sector_diverse():
    symbols = load_market_universe()
    manifest = json.loads(DEFAULT_UNIVERSE_PATH.read_text(encoding="utf-8"))

    assert manifest["version"] == "us-large-cap-v1"
    assert manifest["selected_as_of"] == "2026-08-01"
    assert symbols == tuple(item["symbol"] for item in manifest["instruments"])
    assert len(symbols) == len(set(symbols)) == 50
    sector_counts: dict[str, int] = {}
    for instrument in manifest["instruments"]:
        sector = instrument["sector"]
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    assert len(sector_counts) == 11
    assert min(sector_counts.values()) >= 3
    assert sector_counts["Information Technology"] > sector_counts["Utilities"]


@pytest.mark.parametrize(
    "manifest, message",
    [
        ({"version": "v1", "instruments": []}, "non-empty"),
        (
            {
                "version": "v1",
                "instruments": [
                    {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                    {"symbol": "aapl", "name": "Apple", "sector": "Technology"},
                ],
            },
            "duplicate",
        ),
        (
            {
                "version": "v1",
                "instruments": [{"symbol": "INVALID SYMBOL", "name": "Invalid", "sector": "Other"}],
            },
            "symbol",
        ),
    ],
)
def test_market_universe_rejects_invalid_manifests(tmp_path, manifest, message):
    path = tmp_path / "universe.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MarketUniverseError, match=message):
        load_market_universe(path)
