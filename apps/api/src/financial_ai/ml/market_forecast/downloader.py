import argparse
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from financial_ai.market_data_service import MarketDataProvider, get_market_data_provider
from financial_ai.ml.market_forecast.daily_bars import (
    DAILY_BAR_COLUMNS,
    validate_daily_bars,
)
from financial_ai.ml.market_forecast.snapshot import (
    DEFAULT_OUTPUT_DIRECTORY,
    write_market_snapshot,
)
from financial_ai.ml.market_forecast.universe import load_market_universe


def normalize_symbol_universe(symbols: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(symbol.strip().upper() for symbol in symbols)
    if not normalized:
        raise ValueError("At least one market symbol is required")
    if any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,31}", symbol) for symbol in normalized):
        raise ValueError("Market symbols must use valid uppercase ticker notation")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Market symbol universe contains duplicates")
    return normalized


def resolve_download_symbols(
    symbols: str | None,
    universe_path: Path | None,
) -> tuple[str, ...]:
    """Resolve symbols from either an inline list or a versioned universe manifest."""
    if (symbols is None) == (universe_path is None):
        raise ValueError("Provide either symbols or a universe path")
    if universe_path is not None:
        return load_market_universe(universe_path)
    return normalize_symbol_universe(symbols.split(","))


def download_daily_bars(
    provider: MarketDataProvider,
    symbols: Sequence[str],
    date_from: date,
    date_to: date,
) -> pd.DataFrame:
    """Download and validate daily OHLCV observations for an explicit universe."""
    if date_from > date_to:
        raise ValueError("date_from must be earlier than or equal to date_to")

    normalized_symbols = normalize_symbol_universe(symbols)
    rows: list[dict[str, object]] = []
    for symbol in normalized_symbols:
        observations = provider.history(
            symbol,
            date_from=date_from,
            date_to=date_to,
        )
        for observation in observations:
            rows.append(
                {
                    "symbol": symbol,
                    "observed_on": observation.observed_on,
                    "open": observation.open,
                    "high": observation.high,
                    "low": observation.low,
                    "close": observation.close,
                    "adjusted_close": observation.adjusted_close,
                    "volume": observation.volume,
                }
            )

    frame = pd.DataFrame(rows, columns=DAILY_BAR_COLUMNS)
    return validate_daily_bars(frame)


def download_market_snapshot(
    symbols: Sequence[str],
    date_from: date,
    date_to: date,
    snapshot_version: str,
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    provider: MarketDataProvider | None = None,
) -> tuple[Path, Path]:
    active_provider = provider or get_market_data_provider("external", historical_feed="sip")
    frame = download_daily_bars(active_provider, symbols, date_from, date_to)
    feed = getattr(active_provider, "historical_feed", active_provider.name)
    return write_market_snapshot(
        frame,
        snapshot_version,
        provider=active_provider.name,
        feed=feed,
        output_directory=output_directory,
    )


def run() -> None:
    parser = argparse.ArgumentParser(description="Download and freeze daily market observations")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--symbols", help="Comma-separated ticker symbols")
    source.add_argument("--universe-path", type=Path, help="Versioned universe manifest")
    parser.add_argument("--date-from", type=date.fromisoformat, required=True)
    parser.add_argument("--date-to", type=date.fromisoformat, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    symbols = resolve_download_symbols(args.symbols, args.universe_path)
    csv_path, metadata_path = download_market_snapshot(
        symbols,
        args.date_from,
        args.date_to,
        args.version,
        output_directory=args.output_directory,
    )
    print(f"Market snapshot: {csv_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    run()
