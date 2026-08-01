from decimal import Decimal


def discover(client, query: str, symbol: str) -> dict[str, object]:
    response = client.get("/v1/market/instruments", params={"query": query})
    assert response.status_code == 200
    return next(item for item in response.json() if item["symbol"] == symbol)


def create_portfolio(client, cash: str = "10000.00", currency: str = "EUR") -> dict[str, object]:
    response = client.post(
        "/v1/paper-portfolios",
        json={"name": "Paper Portfolio", "starting_cash": cash, "base_currency": currency},
    )
    assert response.status_code == 201
    return response.json()


def order(
    client,
    portfolio_id: str,
    instrument_id: str,
    side: str,
    quantity: str,
    client_order_id: str,
):
    return client.post(
        f"/v1/paper-portfolios/{portfolio_id}/orders",
        json={
            "client_order_id": client_order_id,
            "instrument_id": instrument_id,
            "side": side,
            "quantity": quantity,
        },
    )


def test_paper_buy_uses_server_quote_and_derives_cash_holdings_and_pnl(client):
    instrument = discover(client, "world", "WORLD-ETF")
    portfolio = create_portfolio(client)
    executed = order(client, portfolio["id"], instrument["id"], "buy", "10", "buy-1")
    detail = client.get(f"/v1/paper-portfolios/{portfolio['id']}")

    assert executed.status_code == 201
    assert executed.json()["price_source"] == "demo"
    assert detail.status_code == 200
    payload = detail.json()
    execution_price = Decimal(executed.json()["unit_price"])
    expected_cash = (Decimal("10000.00") - execution_price * 10).quantize(Decimal("0.01"))
    assert Decimal(payload["cash_balance"]) == expected_cash
    assert payload["trade_count"] == 1
    assert payload["holdings"][0]["instrument"]["symbol"] == "WORLD-ETF"
    assert Decimal(payload["holdings"][0]["quantity"]) == Decimal("10")
    assert Decimal(payload["holdings"][0]["unrealized_pnl"]) == Decimal("0.00")
    assert payload["warnings"][0] == "Paper trading only: no real order is placed."


def test_partial_sale_updates_average_cost_and_realized_pnl(client):
    instrument = discover(client, "world", "WORLD-ETF")
    portfolio = create_portfolio(client)
    order(client, portfolio["id"], instrument["id"], "buy", "10", "buy-1")
    sale = order(client, portfolio["id"], instrument["id"], "sell", "4", "sell-1")
    detail = client.get(f"/v1/paper-portfolios/{portfolio['id']}").json()

    assert sale.status_code == 201
    assert Decimal(detail["holdings"][0]["quantity"]) == Decimal("6")
    assert Decimal(detail["realized_pnl"]) == Decimal("0.00")
    assert Decimal(detail["total_pnl"]) == Decimal("0.00")


def test_paper_orders_reject_insufficient_cash_holdings_and_currency_mismatch(client):
    eur = discover(client, "world", "WORLD-ETF")
    usd = discover(client, "US Technology Demo A", "US-TECH-A")
    portfolio = create_portfolio(client, cash="100.00")

    too_expensive = order(client, portfolio["id"], eur["id"], "buy", "100", "buy-1")
    short_sale = order(client, portfolio["id"], eur["id"], "sell", "1", "sell-1")
    wrong_currency = order(client, portfolio["id"], usd["id"], "buy", "1", "buy-2")

    assert too_expensive.status_code == 409
    assert too_expensive.json()["detail"]["code"] == "insufficient_paper_cash"
    assert short_sale.status_code == 409
    assert short_sale.json()["detail"]["code"] == "insufficient_paper_holdings"
    assert wrong_currency.status_code == 409
    assert wrong_currency.json()["detail"]["code"] == "paper_currency_mismatch"


def test_client_order_id_is_idempotent_but_cannot_be_reused_for_different_order(client):
    instrument = discover(client, "world", "WORLD-ETF")
    portfolio = create_portfolio(client)

    first = order(client, portfolio["id"], instrument["id"], "buy", "1", "stable-order")
    replay = order(client, portfolio["id"], instrument["id"], "buy", "1", "stable-order")
    conflict = order(client, portfolio["id"], instrument["id"], "buy", "2", "stable-order")
    detail = client.get(f"/v1/paper-portfolios/{portfolio['id']}").json()

    assert replay.status_code == 201
    assert replay.json()["id"] == first.json()["id"]
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "paper_order_idempotency_conflict"
    assert detail["trade_count"] == 1


def test_order_contract_forbids_browser_supplied_execution_price(client):
    instrument = discover(client, "world", "WORLD-ETF")
    portfolio = create_portfolio(client)
    response = client.post(
        f"/v1/paper-portfolios/{portfolio['id']}/orders",
        json={
            "client_order_id": "unsafe-price",
            "instrument_id": instrument["id"],
            "side": "buy",
            "quantity": "1",
            "unit_price": "0.01",
        },
    )

    assert response.status_code == 422
