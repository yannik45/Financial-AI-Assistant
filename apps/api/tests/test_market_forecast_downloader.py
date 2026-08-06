from datetime import date
from decimal import Decimal

import pandas as pd
import pytest
from financial_ai.market_data_service import ProviderInstrument, ProviderPrice
from financial_ai.ml.market_forecast.downloader import (
    download_daily_bars,
    normalize_symbol_universe,
    resolve_download_symbols,
)


class RecordingHistoryProvider:
    name = "recording"

    def __init__(self, missing_ohlc: bool = False) -> None:
        self.calls: list[tuple[str, date | None, date | None]] = []
        self.missing_ohlc = missing_ohlc

    def search(self, query: str, limit: int) -> list[ProviderInstrument]:
        return []

    def history(
        self,
        symbol: str,
        exchange: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[ProviderPrice]:
        self.calls.append((symbol, date_from, date_to))
        return [
            ProviderPrice(
                observed_on=date(2024, 1, 2),
                open=None if self.missing_ohlc else Decimal("100"),
                high=Decimal("103"),
                low=Decimal("99"),
                close=Decimal("102"),
                adjusted_close=Decimal("102"),
                volume=Decimal("1000000"),
            )
        ]


def test_symbol_universe_is_explicit_normalized_and_unique():
    assert normalize_symbol_universe([" aapl ", "MSFT"]) == ("AAPL", "MSFT")
    with pytest.raises(ValueError, match="duplicates"):
        normalize_symbol_universe(["AAPL", "aapl"])
    with pytest.raises(ValueError, match="valid uppercase"):
        normalize_symbol_universe(["invalid ticker"])


def test_download_symbols_can_be_loaded_from_versioned_manifest(tmp_path):
    manifest_path = tmp_path / "universe.json"
    manifest_path.write_text(
        '{"version":"v1","instruments":['
        '{"symbol":"AAPL","name":"Apple","sector":"Technology"}]}'
    )

    assert resolve_download_symbols(None, manifest_path) == ("AAPL",)
    assert resolve_download_symbols(" msft,AAPL ", None) == ("MSFT", "AAPL")
    with pytest.raises(ValueError, match="either"):
        resolve_download_symbols(None, None)


def test_download_daily_bars_maps_provider_observations_and_validates_them():
    provider = RecordingHistoryProvider()

    result = download_daily_bars(
        provider,
        [" msft ", "AAPL"],
        date(2024, 1, 1),
        date(2024, 1, 31),
    )

    assert provider.calls == [
        ("MSFT", date(2024, 1, 1), date(2024, 1, 31)),
        ("AAPL", date(2024, 1, 1), date(2024, 1, 31)),
    ]
    assert result["symbol"].tolist() == ["AAPL", "MSFT"]
    assert pd.api.types.is_datetime64_any_dtype(result["observed_on"])
    assert result.loc[0, "open"] == 100
    assert result.loc[0, "volume"] == 1000000


def test_download_daily_bars_rejects_invalid_dates_and_incomplete_ohlc():
    with pytest.raises(ValueError, match="date_from"):
        download_daily_bars(
            RecordingHistoryProvider(),
            ["AAPL"],
            date(2024, 2, 1),
            date(2024, 1, 1),
        )

    with pytest.raises(ValueError, match="numeric"):
        download_daily_bars(
            RecordingHistoryProvider(missing_ohlc=True),
            ["AAPL"],
            date(2024, 1, 1),
            date(2024, 1, 31),
        )
