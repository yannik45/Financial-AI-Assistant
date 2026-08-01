from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Protocol

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from financial_ai.config import get_settings
from financial_ai.market_data import ASSETS, DemoMarketDataProvider
from financial_ai.models import MarketInstrument, MarketPriceObservation
from financial_ai.schemas import (
    MarketHistoryRead,
    MarketInstrumentRead,
    MarketPriceRead,
    MarketQuoteRead,
)


class MarketDataProviderError(RuntimeError):
    """Raised when a configured external market-data provider cannot satisfy a request."""


class InstrumentNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class ProviderInstrument:
    symbol: str
    name: str
    exchange: str | None
    currency: str
    asset_class: str
    region: str | None = None


@dataclass(frozen=True)
class ProviderPrice:
    observed_on: date
    close: Decimal
    adjusted_close: Decimal | None = None
    volume: Decimal | None = None


class MarketDataProvider(Protocol):
    name: str

    def search(self, query: str, limit: int) -> list[ProviderInstrument]: ...

    def history(
        self,
        symbol: str,
        exchange: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[ProviderPrice]: ...


class DemoProviderAdapter:
    name = "demo"

    def __init__(self, provider: DemoMarketDataProvider | None = None) -> None:
        self._provider = provider or DemoMarketDataProvider()

    def search(self, query: str, limit: int) -> list[ProviderInstrument]:
        normalized = query.casefold().strip()
        matches = [
            asset
            for asset in self._provider.catalog()
            if normalized in asset.symbol.casefold() or normalized in asset.name.casefold()
        ]
        return [
            ProviderInstrument(
                symbol=asset.symbol,
                name=asset.name,
                exchange="DEMO",
                currency=asset.currency,
                asset_class=asset.asset_class,
                region=asset.region,
            )
            for asset in matches[:limit]
        ]

    def history(
        self,
        symbol: str,
        exchange: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[ProviderPrice]:
        normalized = symbol.upper()
        if normalized not in ASSETS:
            raise InstrumentNotFoundError(f"Unknown demo instrument: {normalized}")
        series = self._provider.prices([normalized])[normalized]
        if date_from:
            series = series.loc[str(date_from) :]
        if date_to:
            series = series.loc[: str(date_to)]
        return [
            ProviderPrice(observed_on=index.date(), close=Decimal(str(round(value, 8))))
            for index, value in series.items()
        ]


class TwelveDataProvider:
    name = "twelve_data"
    base_url = "https://api.twelvedata.com"

    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        if not api_key.strip():
            raise ValueError("Twelve Data requires a non-empty API key")
        self._api_key = api_key
        self._client = client or httpx.Client(base_url=self.base_url, timeout=15.0)

    def _get(self, path: str, params: dict[str, str | int]) -> dict[str, object]:
        try:
            response = self._client.get(path, params={**params, "apikey": self._api_key})
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MarketDataProviderError("Twelve Data request failed") from exc
        if not isinstance(payload, dict):
            raise MarketDataProviderError("Twelve Data returned an invalid response")
        if payload.get("status") == "error" or "code" in payload:
            message = str(payload.get("message", "Twelve Data rejected the request"))
            raise MarketDataProviderError(message)
        return payload

    def search(self, query: str, limit: int) -> list[ProviderInstrument]:
        payload = self._get("/symbol_search", {"symbol": query, "outputsize": limit})
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise MarketDataProviderError("Twelve Data search response is invalid")
        results: list[ProviderInstrument] = []
        for item in data[:limit]:
            if not isinstance(item, dict) or not item.get("symbol") or not item.get("currency"):
                continue
            results.append(
                ProviderInstrument(
                    symbol=str(item["symbol"]).upper(),
                    name=str(item.get("instrument_name") or item["symbol"]),
                    exchange=str(item["exchange"]) if item.get("exchange") else None,
                    currency=str(item["currency"]).upper(),
                    asset_class=str(item.get("instrument_type") or "Equity"),
                    region=str(item["country"]) if item.get("country") else None,
                )
            )
        return results

    def history(
        self,
        symbol: str,
        exchange: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[ProviderPrice]:
        params: dict[str, str | int] = {
            "symbol": symbol.upper(),
            "interval": "1day",
            "order": "ASC",
            "outputsize": 5000,
            "timezone": "UTC",
            "adjustment": "all",
        }
        if exchange:
            params["exchange"] = exchange
        if date_from:
            params["start_date"] = date_from.isoformat()
        if date_to:
            params["end_date"] = date_to.isoformat()
        payload = self._get("/time_series", params)
        values = payload.get("values", [])
        if not isinstance(values, list):
            raise MarketDataProviderError("Twelve Data history response is invalid")
        points: list[ProviderPrice] = []
        for item in values:
            if not isinstance(item, dict) or not item.get("datetime") or not item.get("close"):
                continue
            points.append(
                ProviderPrice(
                    observed_on=date.fromisoformat(str(item["datetime"])[:10]),
                    close=Decimal(str(item["close"])),
                    volume=Decimal(str(item["volume"])) if item.get("volume") else None,
                )
            )
        if not points:
            raise InstrumentNotFoundError(f"No daily prices found for {symbol.upper()}")
        return points


@lru_cache
def get_market_data_provider() -> MarketDataProvider:
    settings = get_settings()
    if settings.market_data_provider == "demo":
        return DemoProviderAdapter()
    if not settings.market_data_api_key:
        raise RuntimeError(
            "FINANCIAL_AI_MARKET_DATA_API_KEY is required when Twelve Data is configured"
        )
    return TwelveDataProvider(settings.market_data_api_key)


ProviderFactory = Callable[[], MarketDataProvider]


class MarketDataService:
    def __init__(
        self, session: Session, provider: MarketDataProvider, cache_hours: int = 24
    ) -> None:
        self._session = session
        self._provider = provider
        self._cache_ttl = timedelta(hours=cache_hours)

    def search(self, query: str, limit: int = 10) -> list[MarketInstrument]:
        discovered = self._provider.search(query, limit)
        instruments = [self._upsert_instrument(item) for item in discovered]
        self._session.commit()
        return instruments

    def quote(self, instrument_id: str, refresh: bool = False) -> MarketQuoteRead:
        instrument = self._get_instrument(instrument_id)
        latest = self._latest(instrument.id)
        if refresh or latest is None or self._is_stale(latest.retrieved_at):
            self._refresh(instrument)
            latest = self._latest(instrument.id)
        if latest is None:
            raise InstrumentNotFoundError(f"No price is available for {instrument.symbol}")
        return MarketQuoteRead(
            instrument=MarketInstrumentRead.model_validate(instrument),
            observed_on=latest.observed_on,
            close=latest.close,
            adjusted_close=latest.adjusted_close,
            volume=latest.volume,
            source=latest.source,
            retrieved_at=latest.retrieved_at,
            is_stale=self._is_stale(latest.retrieved_at),
        )

    def history(
        self,
        instrument_id: str,
        date_from: date | None = None,
        date_to: date | None = None,
        refresh: bool = False,
    ) -> MarketHistoryRead:
        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from must not exceed date_to")
        instrument = self._get_instrument(instrument_id)
        statement = self._history_statement(instrument.id, date_from, date_to)
        cached = list(self._session.scalars(statement).all())
        if refresh or not cached:
            self._refresh(instrument, date_from, date_to)
            cached = list(self._session.scalars(statement).all())
        if not cached:
            raise InstrumentNotFoundError(f"No price history is available for {instrument.symbol}")
        return MarketHistoryRead(
            instrument=MarketInstrumentRead.model_validate(instrument),
            source=cached[-1].source,
            retrieved_at=max(item.retrieved_at for item in cached),
            points=[
                MarketPriceRead(
                    observed_on=item.observed_on,
                    close=item.close,
                    adjusted_close=item.adjusted_close,
                    volume=item.volume,
                )
                for item in cached
            ],
        )

    def _get_instrument(self, instrument_id: str) -> MarketInstrument:
        instrument = self._session.get(MarketInstrument, instrument_id)
        if instrument is None or instrument.provider != self._provider.name:
            raise InstrumentNotFoundError(f"Unknown instrument: {instrument_id}")
        return instrument

    def _upsert_instrument(self, item: ProviderInstrument) -> MarketInstrument:
        normalized = item.symbol.upper()
        instrument = self._session.scalar(
            select(MarketInstrument).where(
                MarketInstrument.provider == self._provider.name,
                MarketInstrument.symbol == normalized,
                MarketInstrument.exchange == (item.exchange or ""),
            )
        )
        now = datetime.now(UTC).replace(tzinfo=None)
        if instrument is None:
            instrument = MarketInstrument(provider=self._provider.name, symbol=normalized)
            self._session.add(instrument)
        instrument.name = item.name
        instrument.exchange = item.exchange or ""
        instrument.currency = item.currency.upper()
        instrument.asset_class = item.asset_class
        instrument.region = item.region
        instrument.is_active = True
        instrument.updated_at = now
        self._session.flush()
        return instrument

    def _refresh(
        self,
        instrument: MarketInstrument,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> None:
        retrieved_at = datetime.now(UTC).replace(tzinfo=None)
        for point in self._provider.history(
            instrument.symbol, instrument.exchange or None, date_from, date_to
        ):
            observation = self._session.scalar(
                select(MarketPriceObservation).where(
                    MarketPriceObservation.instrument_id == instrument.id,
                    MarketPriceObservation.observed_on == point.observed_on,
                )
            )
            if observation is None:
                observation = MarketPriceObservation(
                    instrument_id=instrument.id, observed_on=point.observed_on
                )
                self._session.add(observation)
            observation.close = point.close
            observation.adjusted_close = point.adjusted_close
            observation.volume = point.volume
            observation.source = self._provider.name
            observation.retrieved_at = retrieved_at
        self._session.commit()

    def _latest(self, instrument_id: str) -> MarketPriceObservation | None:
        return self._session.scalar(
            select(MarketPriceObservation)
            .where(MarketPriceObservation.instrument_id == instrument_id)
            .order_by(MarketPriceObservation.observed_on.desc())
            .limit(1)
        )

    @staticmethod
    def _history_statement(instrument_id: str, date_from: date | None, date_to: date | None):
        statement = select(MarketPriceObservation).where(
            MarketPriceObservation.instrument_id == instrument_id
        )
        if date_from:
            statement = statement.where(MarketPriceObservation.observed_on >= date_from)
        if date_to:
            statement = statement.where(MarketPriceObservation.observed_on <= date_to)
        return statement.order_by(MarketPriceObservation.observed_on)

    def _is_stale(self, retrieved_at: datetime) -> bool:
        return datetime.now(UTC).replace(tzinfo=None) - retrieved_at > self._cache_ttl


def build_market_data_service(session: Session) -> MarketDataService:
    settings = get_settings()
    return MarketDataService(
        session=session,
        provider=get_market_data_provider(),
        cache_hours=settings.market_data_cache_hours,
    )
