"""Versioned instrument-universe loading for market forecast experiments."""

import json
import re
from pathlib import Path

DEFAULT_UNIVERSE_PATH = Path("data/market/market_forecast_universe_v1.json")


class MarketUniverseError(ValueError):
    """Raised when a market forecast universe violates its data contract."""


def load_market_universe(path: Path = DEFAULT_UNIVERSE_PATH) -> tuple[str, ...]:
    """Load an ordered, validated symbol universe from a versioned manifest."""
    manifest = _read_manifest(path)
    if not isinstance(manifest, dict):
        raise MarketUniverseError("Market universe must be a JSON object")

    version = manifest.get("version")
    if not isinstance(version, str) or not version.strip():
        raise MarketUniverseError("Market universe version must be a non-empty string")

    instruments = manifest.get("instruments")
    if not isinstance(instruments, list) or not instruments:
        raise MarketUniverseError("Market universe instruments must be a non-empty list")

    symbols: list[str] = []
    seen_symbols: set[str] = set()
    for instrument in instruments:
        if not isinstance(instrument, dict):
            raise MarketUniverseError("Market universe instruments must be JSON objects")

        for field in ("symbol", "name", "sector"):
            value = instrument.get(field)
            if not isinstance(value, str) or not value.strip():
                raise MarketUniverseError(
                    f"Market universe instrument {field} must be a non-empty string"
                )

        symbol = instrument["symbol"].strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,31}", symbol):
            raise MarketUniverseError(f"Invalid market universe symbol: {symbol}")
        if symbol in seen_symbols:
            raise MarketUniverseError(f"Market universe contains duplicate symbol: {symbol}")

        seen_symbols.add(symbol)
        symbols.append(symbol)

    return tuple(symbols)


def _read_manifest(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarketUniverseError(f"Unable to read market universe: {path}") from exc
