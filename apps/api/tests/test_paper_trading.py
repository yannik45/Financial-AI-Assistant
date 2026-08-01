from decimal import Decimal

from financial_ai.clock import business_today


def discover(client, query: str, symbol: str) -> dict[str, object]:
    response = client.get("/v1/market/instruments", params={"query": query})
    assert response.status_code == 200
    return next(item for item in response.json() if item["symbol"] == symbol)


def create_portfolio(client, cash: str = "10000.00", currency: str = "EUR") -> dict[str, object]:
    response = client.post(
        "/v1/portfolios",
        json={"name": "Trading Portfolio", "starting_cash": cash, "base_currency": currency},
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
        f"/v1/portfolios/{portfolio_id}/orders",
        json={
            "client_order_id": client_order_id,
            "instrument_id": instrument_id,
            "side": side,
            "quantity": quantity,
        },
    )


def test_buy_uses_server_quote_and_updates_cash_holdings_and_transaction_ledger(client):
    instrument = discover(client, "world", "WORLD-ETF")
    portfolio = create_portfolio(client)
    executed = order(client, portfolio["id"], instrument["id"], "buy", "10", "buy-1")
    detail = client.get(f"/v1/portfolios/{portfolio['id']}/overview")

    assert executed.status_code == 201
    assert executed.json()["price_source"] == "demo"
    assert executed.json()["booked_at"] == business_today().isoformat()
    assert executed.json()["price_observed_on"] == "2026-06-30"
    assert detail.status_code == 200
    payload = detail.json()
    execution_price = Decimal(executed.json()["unit_price"])
    expected_cash = (Decimal("10000.00") - execution_price * 10).quantize(Decimal("0.01"))
    assert Decimal(payload["cash_balance"]) == expected_cash
    assert payload["trade_count"] == 1
    assert payload["holdings"][0]["instrument"]["symbol"] == "WORLD-ETF"
    assert Decimal(payload["holdings"][0]["quantity"]) == Decimal("10")
    assert Decimal(payload["holdings"][0]["unrealized_pnl"]) == Decimal("0.00")
    transactions = client.get("/v1/transactions", params={"account_id": portfolio["id"]})
    assert transactions.status_code == 404  # Portfolio IDs are not account IDs.
    accounts = client.get("/v1/accounts").json()
    brokerage = next(item for item in accounts if item["name"] == "Trading Portfolio Brokerage")
    ledger = client.get("/v1/transactions", params={"account_id": brokerage["id"]}).json()
    assert ledger["items"][0]["transaction_type"] == "security_buy"
    assert ledger["items"][0]["booked_at"] == business_today().isoformat()
    assert ledger["items"][0]["price_observed_on"] == "2026-06-30"
    assert Decimal(ledger["items"][0]["amount"]) == (-execution_price * 10).quantize(
        Decimal("0.01")
    )
    assert Decimal(brokerage["current_balance"]) == expected_cash


def test_partial_sale_updates_average_cost_and_realized_pnl(client):
    instrument = discover(client, "world", "WORLD-ETF")
    portfolio = create_portfolio(client)
    order(client, portfolio["id"], instrument["id"], "buy", "10", "buy-1")
    sale = order(client, portfolio["id"], instrument["id"], "sell", "4", "sell-1")
    detail = client.get(f"/v1/portfolios/{portfolio['id']}/overview").json()

    assert sale.status_code == 201
    assert Decimal(detail["holdings"][0]["quantity"]) == Decimal("6")
    assert Decimal(detail["realized_pnl"]) == Decimal("0.00")
    assert Decimal(detail["total_pnl"]) == Decimal("0.00")


def test_orders_reject_insufficient_cash_and_holdings_but_convert_foreign_currency(client):
    eur = discover(client, "world", "WORLD-ETF")
    usd = discover(client, "US Technology Demo A", "US-TECH-A")
    portfolio = create_portfolio(client, cash="1000.00")

    too_expensive = order(client, portfolio["id"], eur["id"], "buy", "100", "buy-1")
    short_sale = order(client, portfolio["id"], eur["id"], "sell", "1", "sell-1")
    fractional_buy = order(client, portfolio["id"], usd["id"], "buy", "0.1", "fractional")
    foreign_buy = order(client, portfolio["id"], usd["id"], "buy", "1", "buy-2")
    foreign_sell = order(client, portfolio["id"], usd["id"], "sell", "1", "sell-2")

    assert too_expensive.status_code == 409
    assert too_expensive.json()["detail"]["code"] == "insufficient_cash"
    assert short_sale.status_code == 409
    assert short_sale.json()["detail"]["code"] == "insufficient_holdings"
    assert fractional_buy.status_code == 422
    assert foreign_buy.status_code == 201
    assert foreign_buy.json()["instrument_currency"] == "USD"
    assert foreign_buy.json()["currency"] == "EUR"
    assert Decimal(foreign_buy.json()["settlement_amount"]) > 0
    assert foreign_sell.status_code == 201
    assert foreign_sell.json()["instrument_currency"] == "USD"


def test_client_order_id_is_idempotent_but_cannot_be_reused_for_different_order(client):
    instrument = discover(client, "world", "WORLD-ETF")
    portfolio = create_portfolio(client)

    first = order(client, portfolio["id"], instrument["id"], "buy", "1", "stable-order")
    replay = order(client, portfolio["id"], instrument["id"], "buy", "1", "stable-order")
    conflict = order(client, portfolio["id"], instrument["id"], "buy", "2", "stable-order")
    detail = client.get(f"/v1/portfolios/{portfolio['id']}/overview").json()

    assert replay.status_code == 201
    assert replay.json()["id"] == first.json()["id"]
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "order_idempotency_conflict"
    assert detail["trade_count"] == 1


def test_order_contract_forbids_browser_supplied_execution_price(client):
    instrument = discover(client, "world", "WORLD-ETF")
    portfolio = create_portfolio(client)
    response = client.post(
        f"/v1/portfolios/{portfolio['id']}/orders",
        json={
            "client_order_id": "unsafe-price",
            "instrument_id": instrument["id"],
            "side": "buy",
            "quantity": "1",
            "unit_price": "0.01",
        },
    )

    assert response.status_code == 422


def test_buy_and_sell_update_portfolio_risk_analytics(client):
    world = discover(client, "world", "WORLD-ETF")
    bonds = discover(client, "bond", "EURO-BOND")
    portfolio = create_portfolio(client)

    assert order(client, portfolio["id"], world["id"], "buy", "10", "buy-world").status_code == 201
    assert order(client, portfolio["id"], bonds["id"], "buy", "10", "buy-bonds").status_code == 201
    diversified = client.get(f"/v1/portfolios/{portfolio['id']}/analytics")

    assert diversified.status_code == 200
    diversified_payload = diversified.json()
    assert {item["symbol"] for item in diversified_payload["positions"]} == {
        "WORLD-ETF",
        "EURO-BOND",
    }
    assert diversified_payload["concentration_hhi"] < 1
    assert diversified_payload["risk_score"]["methodology_version"] == "portfolio-risk-score-v2"

    assert (
        order(client, portfolio["id"], bonds["id"], "sell", "10", "sell-bonds").status_code == 201
    )
    concentrated = client.get(f"/v1/portfolios/{portfolio['id']}/analytics")

    assert concentrated.status_code == 200
    concentrated_payload = concentrated.json()
    assert [item["symbol"] for item in concentrated_payload["positions"]] == ["WORLD-ETF"]
    assert concentrated_payload["concentration_hhi"] == 1
    assert concentrated_payload["largest_position_weight"] == 1
    assert (
        concentrated_payload["risk_score"]["diversification"]["score"]
        < diversified_payload["risk_score"]["diversification"]["score"]
    )
    assert (
        concentrated_payload["annualized_volatility_percent"]
        != diversified_payload["annualized_volatility_percent"]
    )
