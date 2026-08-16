from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from financial_ai.config import get_settings

DATA_VERSION = "demo-market-2026.1"


@dataclass(frozen=True)
class Asset:
    symbol: str
    name: str
    currency: str
    asset_class: str
    sector: str
    region: str
    start_price: float
    annual_return: float
    annual_volatility: float


@dataclass(frozen=True)
class EconomicExposure:
    asset_class: str
    sector: str
    region: str


# Alpaca identifies these instruments by their US listing but does not expose
# fund holdings. Keep the small override set explicit so broad-market funds can
# receive conservative look-through treatment without classifying every ETF as
# diversified.
BROAD_MARKET_FUND_EXPOSURES: dict[str, EconomicExposure] = {
    "ACWI": EconomicExposure("Equity ETF", "Broad Market", "Global"),
    "VT": EconomicExposure("Equity ETF", "Broad Market", "Global"),
    "VTI": EconomicExposure("Equity ETF", "Broad Market", "United States"),
    "SPY": EconomicExposure("Equity ETF", "Broad Market", "United States"),
    "IVV": EconomicExposure("Equity ETF", "Broad Market", "United States"),
    "VOO": EconomicExposure("Equity ETF", "Broad Market", "United States"),
}


ASSETS: dict[str, Asset] = {
    "WORLD-ETF": Asset(
        "WORLD-ETF",
        "Global Equity Demo ETF",
        "EUR",
        "Equity ETF",
        "Broad Market",
        "Global",
        104,
        0.08,
        0.15,
    ),
    "EURO-BOND": Asset(
        "EURO-BOND",
        "Euro Aggregate Bond Demo ETF",
        "EUR",
        "Bond ETF",
        "Fixed Income",
        "Europe",
        98,
        0.025,
        0.045,
    ),
    "EU-TECH": Asset(
        "EU-TECH",
        "European Technology Demo",
        "EUR",
        "Equity",
        "Technology",
        "Europe",
        82,
        0.11,
        0.24,
    ),
    "EU-HEALTH": Asset(
        "EU-HEALTH",
        "European Healthcare Demo",
        "EUR",
        "Equity",
        "Healthcare",
        "Europe",
        76,
        0.065,
        0.14,
    ),
    "US-TECH-A": Asset(
        "US-TECH-A",
        "US Technology Demo A",
        "USD",
        "Equity",
        "Technology",
        "North America",
        145,
        0.13,
        0.28,
    ),
    "US-TECH-B": Asset(
        "US-TECH-B",
        "US Technology Demo B",
        "USD",
        "Equity",
        "Technology",
        "North America",
        118,
        0.12,
        0.25,
    ),
    "US-HEALTH": Asset(
        "US-HEALTH",
        "US Healthcare Demo",
        "USD",
        "Equity",
        "Healthcare",
        "North America",
        92,
        0.07,
        0.16,
    ),
    "UK-DIVIDEND": Asset(
        "UK-DIVIDEND",
        "UK Dividend Demo ETF",
        "GBP",
        "Equity ETF",
        "Broad Market",
        "Europe",
        64,
        0.055,
        0.13,
    ),
    "JP-EQUITY": Asset(
        "JP-EQUITY",
        "Japan Equity Demo ETF",
        "JPY",
        "Equity ETF",
        "Broad Market",
        "Asia Pacific",
        12800,
        0.07,
        0.18,
    ),
    "GOLD-ETC": Asset(
        "GOLD-ETC", "Gold Demo ETC", "USD", "Commodity", "Commodities", "Global", 181, 0.045, 0.17
    ),
}


class MarketDataError(ValueError):
    pass


class DemoMarketDataProvider:
    """Deterministic synthetic prices and bundled ECB-attributed FX fixtures."""

    def __init__(self, ecb_fx_path: Path | None = None) -> None:
        self._ecb_fx_path = ecb_fx_path or get_settings().ecb_fx_path
        self._dates = pd.bdate_range("2024-01-02", "2026-06-30")
        self._prices = self._generate_prices()
        self._fx = self._generate_fx_fixture()

    def catalog(self) -> list[Asset]:
        return list(ASSETS.values())

    def prices(self, symbols: list[str]) -> pd.DataFrame:
        unknown = sorted(set(symbols) - ASSETS.keys())
        if unknown:
            raise MarketDataError(f"Unknown symbols: {', '.join(unknown)}")
        return self._prices[symbols].copy()

    def eur_per_currency(self, currency: str) -> pd.Series:
        if currency not in self._fx:
            raise MarketDataError(f"Unsupported currency: {currency}")
        return self._fx[currency].copy()

    def fx_on_or_before(self, currency: str, value_date: date) -> float:
        series = self.eur_per_currency(currency)
        eligible = series.loc[: pd.Timestamp(value_date)]
        if eligible.empty:
            raise MarketDataError(f"No {currency}/EUR rate on or before {value_date}")
        return float(eligible.iloc[-1])

    def _generate_prices(self) -> pd.DataFrame:
        rng = np.random.default_rng(20260723)
        common = rng.normal(0.00015, 0.006, len(self._dates))
        values: dict[str, np.ndarray] = {}
        for asset in ASSETS.values():
            daily_drift = asset.annual_return / 252
            daily_vol = asset.annual_volatility / np.sqrt(252)
            idiosyncratic = rng.normal(0, daily_vol, len(self._dates))
            returns = daily_drift + 0.35 * common + np.sqrt(1 - 0.35**2) * idiosyncratic
            values[asset.symbol] = asset.start_price * np.exp(np.cumsum(returns))
        return pd.DataFrame(values, index=self._dates)

    def _generate_fx_fixture(self) -> dict[str, pd.Series]:
        snapshot = self._ecb_fx_path
        if not snapshot.exists():
            raise MarketDataError(f"ECB FX snapshot is missing: {snapshot}")
        frame = pd.read_csv(snapshot, usecols=["CURRENCY", "TIME_PERIOD", "OBS_VALUE"])
        frame["TIME_PERIOD"] = pd.to_datetime(frame["TIME_PERIOD"])
        result: dict[str, pd.Series] = {"EUR": pd.Series(1.0, index=self._dates)}
        for currency in ("USD", "GBP", "JPY"):
            quoted_per_eur = frame.loc[frame["CURRENCY"] == currency].set_index("TIME_PERIOD")[
                "OBS_VALUE"
            ]
            # ECB series are quoted as foreign-currency units per EUR; analytics need EUR per unit.
            result[currency] = (1.0 / quoted_per_eur).reindex(self._dates).ffill().bfill()
        return result


market_data_provider = DemoMarketDataProvider()
