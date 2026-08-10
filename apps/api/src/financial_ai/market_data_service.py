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
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
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
        matches = self._provider.catalog()
        if normalized != "*":
            matches = [
                asset
                for asset in matches
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


class AlpacaProvider:
    """US asset discovery and adjusted daily bars with a keyless SEC search fallback."""

    name = "alpaca"
    data_url = "https://data.alpaca.markets"
    trading_url = "https://paper-api.alpaca.markets"
    sec_url = "https://www.sec.gov"

    def __init__(
        self,
        api_key: str | None,
        secret_key: str | None,
        *,
        data_client: httpx.Client | None = None,
        trading_client: httpx.Client | None = None,
        sec_client: httpx.Client | None = None,
        sec_user_agent: str,
        historical_feed: str = "iex",
    ) -> None:
        self._api_key = api_key.strip() if api_key else None
        self._secret_key = secret_key.strip() if secret_key else None
        if bool(self._api_key) != bool(self._secret_key):
            raise ValueError("Alpaca requires both API key and secret key")
        if historical_feed not in {"iex", "sip"}:
            raise ValueError("Alpaca historical feed must be iex or sip")
        self.historical_feed = historical_feed
        self._data_client = data_client or httpx.Client(base_url=self.data_url, timeout=20.0)
        self._trading_client = trading_client or httpx.Client(
            base_url=self.trading_url, timeout=20.0
        )
        self._sec_client = sec_client or httpx.Client(base_url=self.sec_url, timeout=20.0)
        self._sec_headers = {"User-Agent": sec_user_agent}
        self._catalog: list[ProviderInstrument] | None = None

    @property
    def configured(self) -> bool:
        return bool(self._api_key and self._secret_key)

    @property
    def source_name(self) -> str:
        return f"{self.name}:{self.historical_feed}"

    def _alpaca_headers(self) -> dict[str, str]:
        if not self.configured:
            raise MarketDataProviderError(
                "Alpaca credentials are required for external price history"
            )
        return {
            "APCA-API-KEY-ID": self._api_key or "",
            "APCA-API-SECRET-KEY": self._secret_key or "",
        }

    @staticmethod
    def _payload(response: httpx.Response, provider: str) -> object:
        try:
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            if response.status_code == 429:
                raise MarketDataProviderError(f"{provider} rate limit reached") from exc
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = None
            message = error_payload.get("message") if isinstance(error_payload, dict) else None
            detail = f": {message}" if isinstance(message, str) and message else ""
            raise MarketDataProviderError(f"{provider} request failed{detail}") from exc

    def search(self, query: str, limit: int) -> list[ProviderInstrument]:
        normalized = query.casefold().strip()
        if self._catalog is None:
            self._catalog = (
                self._load_alpaca_catalog() if self.configured else self._load_sec_catalog()
            )
        matches = [
            item
            for item in self._catalog
            if normalized in item.symbol.casefold() or normalized in item.name.casefold()
        ]
        matches.sort(
            key=lambda item: (
                not item.symbol.casefold().startswith(normalized),
                not item.name.casefold().startswith(normalized),
                item.symbol,
            )
        )
        return matches[:limit]

    def _load_alpaca_catalog(self) -> list[ProviderInstrument]:
        try:
            response = self._trading_client.get(
                "/v2/assets",
                params={"status": "active", "asset_class": "us_equity"},
                headers=self._alpaca_headers(),
            )
        except httpx.HTTPError as exc:
            raise MarketDataProviderError("Alpaca asset catalog request failed") from exc
        payload = self._payload(response, "Alpaca")
        if not isinstance(payload, list):
            raise MarketDataProviderError("Alpaca returned an invalid asset catalog")
        results: list[ProviderInstrument] = []
        for item in payload:
            if not isinstance(item, dict) or not item.get("symbol") or not item.get("name"):
                continue
            results.append(
                ProviderInstrument(
                    symbol=str(item["symbol"]).upper(),
                    name=str(item["name"]),
                    exchange=str(item["exchange"]) if item.get("exchange") else None,
                    currency="USD",
                    asset_class="US Equity",
                    region="United States",
                )
            )
        return results

    def _load_sec_catalog(self) -> list[ProviderInstrument]:
        try:
            response = self._sec_client.get(
                "/files/company_tickers.json", headers=self._sec_headers
            )
        except httpx.HTTPError as exc:
            raise MarketDataProviderError("SEC company catalog request failed") from exc
        payload = self._payload(response, "SEC")
        if not isinstance(payload, dict):
            raise MarketDataProviderError("SEC returned an invalid company catalog")
        return [
            ProviderInstrument(
                symbol=str(item["ticker"]).upper(),
                name=str(item["title"]),
                exchange=None,
                currency="USD",
                asset_class="US Equity",
                region="United States",
            )
            for item in payload.values()
            if isinstance(item, dict) and item.get("ticker") and item.get("title")
        ]

    def history(
        self,
        symbol: str,
        exchange: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[ProviderPrice]:
        del exchange
        normalized = symbol.upper()
        today = datetime.now(UTC).date()
        start = date_from or (today - timedelta(days=600))
        end = date_to or today
        params: dict[str, str | int] = {
            "symbols": normalized,
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": "all",
            "feed": self.historical_feed,
            "sort": "asc",
            "limit": 10000,
        }
        points: list[ProviderPrice] = []
        page_token: str | None = None
        while True:
            if page_token:
                params["page_token"] = page_token
            try:
                response = self._data_client.get(
                    "/v2/stocks/bars", params=params, headers=self._alpaca_headers()
                )
            except httpx.HTTPError as exc:
                raise MarketDataProviderError("Alpaca price history request failed") from exc
            payload = self._payload(response, "Alpaca")
            if not isinstance(payload, dict):
                raise MarketDataProviderError("Alpaca returned invalid price history")
            bars_by_symbol = payload.get("bars", {})
            bars = bars_by_symbol.get(normalized, []) if isinstance(bars_by_symbol, dict) else []
            for item in bars:
                if not isinstance(item, dict) or not item.get("t") or item.get("c") is None:
                    continue
                close = Decimal(str(item["c"]))
                points.append(
                    ProviderPrice(
                        observed_on=date.fromisoformat(str(item["t"])[:10]),
                        close=close,
                        open=Decimal(str(item["o"])) if item.get("o") is not None else None,
                        high=Decimal(str(item["h"])) if item.get("h") is not None else None,
                        low=Decimal(str(item["l"])) if item.get("l") is not None else None,
                        adjusted_close=close,
                        volume=Decimal(str(item["v"])) if item.get("v") is not None else None,
                    )
                )
            page_token = str(payload["next_page_token"]) if payload.get("next_page_token") else None
            if not page_token:
                break
        if not points:
            raise InstrumentNotFoundError(f"No daily prices found for {normalized}")
        return points


@lru_cache
def get_market_data_provider(
    mode: str = "demo", *, historical_feed: str = "iex"
) -> MarketDataProvider:
    settings = get_settings()
    if mode == "demo":
        return DemoProviderAdapter()
    if mode != "external":
        raise ValueError(f"Unsupported market-data mode: {mode}")
    return AlpacaProvider(
        settings.alpaca_api_key,
        settings.alpaca_secret_key,
        sec_user_agent=settings.sec_user_agent,
        historical_feed=historical_feed,
    )


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

    def quote(
        self, instrument_id: str, refresh: bool = False, allow_stale: bool = True
    ) -> MarketQuoteRead:
        instrument = self._get_instrument(instrument_id)
        latest = self._latest(instrument.id)
        if refresh or latest is None or self._is_stale(latest.retrieved_at):
            try:
                self._refresh(instrument)
                latest = self._latest(instrument.id)
            except MarketDataProviderError:
                if latest is None or refresh or not allow_stale:
                    raise
        if latest is None:
            raise InstrumentNotFoundError(f"No price is available for {instrument.symbol}")
        return MarketQuoteRead(
            instrument=MarketInstrumentRead.model_validate(instrument),
            observed_on=latest.observed_on,
            close=latest.close,
            open=latest.open,
            high=latest.high,
            low=latest.low,
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
                    open=item.open,
                    high=item.high,
                    low=item.low,
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
            observation.open = point.open
            observation.high = point.high
            observation.low = point.low
            observation.adjusted_close = point.adjusted_close
            observation.volume = point.volume
            observation.source = getattr(
                self._provider,
                "source_name",
                self._provider.name,
            )
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

    def is_stale(self, retrieved_at: datetime) -> bool:
        """Return whether cached provider data has exceeded its configured TTL."""
        return self._is_stale(retrieved_at)


def build_market_data_service(session: Session, mode: str = "demo") -> MarketDataService:
    settings = get_settings()
    return MarketDataService(
        session=session,
        provider=get_market_data_provider(mode),
        cache_hours=settings.market_data_cache_hours,
    )
