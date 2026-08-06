from datetime import date
from decimal import Decimal

import httpx
import pytest
from financial_ai.config import get_settings
from financial_ai.database import SessionLocal
from financial_ai.market_data import ASSETS
from financial_ai.market_data_service import (
    AlpacaProvider,
    DemoProviderAdapter,
    MarketDataProviderError,
    MarketDataService,
    ProviderInstrument,
    ProviderPrice,
    get_market_data_provider,
)
from financial_ai.models import MarketInstrument, MarketPriceObservation


class RecordingProvider:
    name = "recording"

    def __init__(self) -> None:
        self.history_calls = 0

    def search(self, query: str, limit: int) -> list[ProviderInstrument]:
        if "ACME" not in query.upper():
            return []
        return [
            ProviderInstrument(
                symbol="ACME",
                name="Acme Corporation",
                exchange="XNAS",
                currency="USD",
                asset_class="Common Stock",
                region="United States",
            )
        ][:limit]

    def history(
        self,
        symbol: str,
        exchange: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[ProviderPrice]:
        self.history_calls += 1
        points = [
            ProviderPrice(date(2026, 7, 30), Decimal("100.25"), volume=Decimal("1200")),
            ProviderPrice(date(2026, 7, 31), Decimal("101.50"), volume=Decimal("1500")),
        ]
        return [
            point
            for point in points
            if (date_from is None or point.observed_on >= date_from)
            and (date_to is None or point.observed_on <= date_to)
        ]


@pytest.fixture
def without_alpaca_credentials(monkeypatch):
    monkeypatch.setenv("FINANCIAL_AI_ALPACA_API_KEY", "")
    monkeypatch.setenv("FINANCIAL_AI_ALPACA_SECRET_KEY", "")
    get_settings.cache_clear()
    get_market_data_provider.cache_clear()
    yield
    get_settings.cache_clear()
    get_market_data_provider.cache_clear()


class FailingProvider(RecordingProvider):
    def history(self, *args, **kwargs) -> list[ProviderPrice]:
        raise MarketDataProviderError("upstream unavailable")


def test_market_service_persists_instruments_and_reuses_fresh_quotes():
    provider = RecordingProvider()
    with SessionLocal() as session:
        service = MarketDataService(session, provider, cache_hours=24)
        matches = service.search("acme")
        first = service.quote(matches[0].id)
        second = service.quote(matches[0].id)

        assert matches[0].symbol == "ACME"
        assert first.close == Decimal("101.50000000")
        assert first.source == "recording"
        assert first.is_stale is False
        assert second.retrieved_at == first.retrieved_at
        assert provider.history_calls == 1
        assert session.query(MarketInstrument).count() == 1
        assert session.query(MarketPriceObservation).count() == 2


def test_demo_catalog_can_be_listed_without_guessing_a_search_term():
    with SessionLocal() as session:
        service = MarketDataService(session, DemoProviderAdapter())

        instruments = service.search("*", limit=25)

        assert len(instruments) == len(ASSETS)
        assert {item.symbol for item in instruments} == set(ASSETS)


def test_stale_quote_can_be_displayed_but_not_used_for_order_pricing():
    provider = RecordingProvider()
    with SessionLocal() as session:
        service = MarketDataService(session, provider, cache_hours=0)
        instrument = service.search("ACME")[0]
        service.quote(instrument.id)
        service._provider = FailingProvider()

        stale = service.quote(instrument.id)
        assert stale.is_stale is True
        with pytest.raises(MarketDataProviderError, match="upstream unavailable"):
            service.quote(instrument.id, allow_stale=False)


def test_market_history_honors_dates_and_explicit_refresh():
    provider = RecordingProvider()
    with SessionLocal() as session:
        service = MarketDataService(session, provider)
        instrument = service.search("ACME")[0]
        result = service.history(
            instrument.id,
            date_from=date(2026, 7, 31),
            date_to=date(2026, 7, 31),
        )
        refreshed = service.history(instrument.id, refresh=True)

        assert [point.observed_on for point in result.points] == [date(2026, 7, 31)]
        assert len(refreshed.points) == 2
        assert provider.history_calls == 2
        assert session.query(MarketPriceObservation).count() == 2


def test_alpaca_adapter_maps_assets_and_paginated_adjusted_daily_bars():
    def trading_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["APCA-API-KEY-ID"] == "key"
        assert request.headers["APCA-API-SECRET-KEY"] == "secret"
        return httpx.Response(
            200,
            json=[
                {
                    "symbol": "AAPL",
                    "name": "Apple Inc.",
                    "exchange": "NASDAQ",
                    "class": "us_equity",
                }
            ],
        )

    def data_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["APCA-API-KEY-ID"] == "key"
        assert request.url.params["adjustment"] == "all"
        assert request.url.params["feed"] == "iex"
        if not request.url.params.get("page_token"):
            return httpx.Response(
                200,
                json={
                    "bars": {
                        "AAPL": [
                            {
                                "t": "2026-07-30T04:00:00Z",
                                "o": 200.0,
                                "h": 206.0,
                                "l": 199.0,
                                "c": 204.1,
                                "v": 1,
                            }
                        ]
                    },
                    "next_page_token": "next",
                },
            )
        return httpx.Response(
            200,
            json={
                "bars": {
                    "AAPL": [
                        {
                            "t": "2026-07-31T04:00:00Z",
                            "o": 204.0,
                            "h": 207.0,
                            "l": 203.0,
                            "c": 205.12,
                            "v": 42000000,
                        }
                    ]
                },
                "next_page_token": None,
            },
        )

    provider = AlpacaProvider(
        "key",
        "secret",
        data_client=httpx.Client(
            base_url="https://data.alpaca.markets", transport=httpx.MockTransport(data_handler)
        ),
        trading_client=httpx.Client(
            base_url="https://paper-api.alpaca.markets",
            transport=httpx.MockTransport(trading_handler),
        ),
        sec_user_agent="test@example.com",
    )

    result = provider.search("Apple", 5)
    history = provider.history("AAPL")

    assert result[0].symbol == "AAPL"
    assert result[0].exchange == "NASDAQ"
    assert [item.close for item in history] == [Decimal("204.1"), Decimal("205.12")]
    assert history[-1].open == Decimal("204.0")
    assert history[-1].high == Decimal("207.0")
    assert history[-1].low == Decimal("203.0")
    assert history[-1].adjusted_close == Decimal("205.12")
    assert history[-1].volume == Decimal("42000000")


def test_alpaca_adapter_uses_explicit_historical_feed():
    def data_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["feed"] == "sip"
        return httpx.Response(
            200,
            json={
                "bars": {
                    "AAPL": [
                        {
                            "t": "2016-01-04T05:00:00Z",
                            "o": 25.65,
                            "h": 26.34,
                            "l": 25.5,
                            "c": 26.34,
                            "v": 270597600,
                        }
                    ]
                },
                "next_page_token": None,
            },
        )

    provider = AlpacaProvider(
        "key",
        "secret",
        data_client=httpx.Client(
            base_url="https://data.alpaca.markets",
            transport=httpx.MockTransport(data_handler),
        ),
        sec_user_agent="test@example.com",
        historical_feed="sip",
    )

    assert provider.history("AAPL", date_from=date(2016, 1, 1))[0].observed_on == date(
        2016, 1, 4
    )


def test_alpaca_search_falls_back_to_keyless_sec_company_catalog():
    def sec_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["User-Agent"] == "portfolio-test test@example.com"
        return httpx.Response(
            200,
            json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}},
        )

    provider = AlpacaProvider(
        None,
        None,
        sec_client=httpx.Client(
            base_url="https://www.sec.gov", transport=httpx.MockTransport(sec_handler)
        ),
        sec_user_agent="portfolio-test test@example.com",
    )

    assert provider.search("Apple", 5)[0].symbol == "AAPL"
    with pytest.raises(MarketDataProviderError, match="credentials"):
        provider.history("AAPL")


def test_market_endpoints_expose_provenance_and_validation(client):
    search = client.get("/v1/market/instruments", params={"query": "world"})
    instrument_id = search.json()[0]["id"]
    quote = client.get(f"/v1/market/instruments/{instrument_id}/quote")
    history = client.get(
        f"/v1/market/instruments/{instrument_id}/history",
        params={"date_from": "2026-06-01", "date_to": "2026-06-30"},
    )
    invalid = client.get(
        f"/v1/market/instruments/{instrument_id}/history",
        params={"date_from": "2026-07-01", "date_to": "2026-06-01"},
    )

    assert search.status_code == 200
    assert search.json()[0]["provider"] == "demo"
    assert quote.status_code == 200
    assert quote.json()["source"] == "demo"
    assert quote.json()["is_stale"] is False
    assert history.status_code == 200
    assert history.json()["points"]
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["code"] == "invalid_market_data_request"


def test_market_status_and_portfolio_mode_default_are_explicit(client, without_alpaca_credentials):
    status = client.get("/v1/market/status")
    created = client.post(
        "/v1/portfolios",
        json={"name": "Demo mode", "starting_cash": "1000", "base_currency": "EUR"},
    )

    assert status.status_code == 200
    assert status.json() == {
        "demo_available": True,
        "external_available": False,
        "external_provider": "alpaca",
    }
    assert created.status_code == 201
    assert created.json()["market_data_mode"] == "demo"


def test_external_portfolio_requires_server_side_api_key(client, without_alpaca_credentials):
    response = client.post(
        "/v1/portfolios",
        json={
            "name": "External mode",
            "starting_cash": "1000",
            "base_currency": "EUR",
            "market_data_mode": "external",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "portfolio_trading_error"
