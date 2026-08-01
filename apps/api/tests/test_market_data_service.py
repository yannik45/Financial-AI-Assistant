from datetime import date
from decimal import Decimal

import httpx
from financial_ai.database import SessionLocal
from financial_ai.market_data_service import (
    MarketDataService,
    ProviderInstrument,
    ProviderPrice,
    TwelveDataProvider,
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


def test_twelve_data_adapter_maps_search_and_daily_history():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["apikey"] == "secret"
        if request.url.path == "/symbol_search":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "symbol": "AAPL",
                            "instrument_name": "Apple Inc",
                            "exchange": "NASDAQ",
                            "currency": "USD",
                            "instrument_type": "Common Stock",
                            "country": "United States",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "values": [
                    {
                        "datetime": "2026-07-31",
                        "close": "205.12",
                        "volume": "42000000",
                    }
                ]
            },
        )

    client = httpx.Client(
        base_url="https://api.twelvedata.com", transport=httpx.MockTransport(handler)
    )
    provider = TwelveDataProvider("secret", client)

    result = provider.search("Apple", 5)
    history = provider.history("AAPL")

    assert result[0].symbol == "AAPL"
    assert result[0].exchange == "NASDAQ"
    assert history[0].close == Decimal("205.12")
    assert history[0].volume == Decimal("42000000")


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
