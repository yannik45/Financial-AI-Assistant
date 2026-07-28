def test_demo_accounts_and_transactions_are_seeded(client):
    accounts_response = client.get("/v1/accounts")
    assert accounts_response.status_code == 200
    accounts = accounts_response.json()
    assert len(accounts) == 3
    assert {account["account_type"] for account in accounts} == {
        "checking",
        "savings",
        "brokerage",
    }
    assert sum(account["transaction_count"] for account in accounts) == 18

    transactions_response = client.get("/v1/transactions")
    assert transactions_response.status_code == 200
    page = transactions_response.json()
    assert page["total"] == 18
    assert len(page["items"]) == 18
    assert page["items"][0]["booked_at"] >= page["items"][-1]["booked_at"]


def test_transactions_can_be_filtered_and_paginated(client):
    accounts = client.get("/v1/accounts").json()
    checking_id = next(
        account["id"] for account in accounts if account["account_type"] == "checking"
    )

    response = client.get(
        "/v1/transactions",
        params={
            "account_id": checking_id,
            "category": "groceries",
            "date_from": "2026-01-01",
            "date_to": "2026-03-31",
            "limit": 1,
        },
    )
    assert response.status_code == 200
    page = response.json()
    assert page["total"] == 2
    assert len(page["items"]) == 1
    assert page["items"][0]["category"] == "Groceries"

    invalid_range = client.get(
        "/v1/transactions", params={"date_from": "2026-04-01", "date_to": "2026-03-01"}
    )
    assert invalid_range.status_code == 422
    assert invalid_range.json()["detail"]["code"] == "invalid_date_range"


def test_manual_bank_transaction_can_be_created(client):
    checking = next(
        account
        for account in client.get("/v1/accounts").json()
        if account["account_type"] == "checking"
    )
    response = client.post(
        "/v1/transactions",
        json={
            "account_id": checking["id"],
            "booked_at": "2026-04-03",
            "name": "Demo coffee",
            "amount": "-4.50",
            "currency": "eur",
            "transaction_type": "card_payment",
            "counterparty": "Coffee Demo",
            "category": "Dining",
            "notes": "Created manually during local testing",
        },
    )
    assert response.status_code == 201
    transaction = response.json()
    assert transaction["source"] == "manual"
    assert transaction["currency"] == "EUR"
    assert transaction["amount"] == "-4.50"
    assert client.get(f"/v1/transactions/{transaction['id']}").status_code == 200
    assert client.get("/v1/transactions").json()["total"] == 19


def test_security_transaction_requires_complete_fields_and_brokerage_account(client):
    accounts = client.get("/v1/accounts").json()
    checking_id = next(
        account["id"] for account in accounts if account["account_type"] == "checking"
    )
    brokerage_id = next(
        account["id"] for account in accounts if account["account_type"] == "brokerage"
    )
    base_payload = {
        "booked_at": "2026-04-04",
        "name": "Buy demo ETF",
        "amount": "-505.00",
        "currency": "EUR",
        "transaction_type": "security_buy",
    }

    missing_fields = client.post(
        "/v1/transactions", json={**base_payload, "account_id": brokerage_id}
    )
    assert missing_fields.status_code == 422

    wrong_account = client.post(
        "/v1/transactions",
        json={
            **base_payload,
            "account_id": checking_id,
            "security_symbol": "world-etf",
            "quantity": "5",
            "unit_price": "100",
        },
    )
    assert wrong_account.status_code == 422
    assert wrong_account.json()["detail"]["code"] == "invalid_account_type"

    valid = client.post(
        "/v1/transactions",
        json={
            **base_payload,
            "account_id": brokerage_id,
            "security_symbol": "world-etf",
            "quantity": "5",
            "unit_price": "100",
            "fees": "5",
        },
    )
    assert valid.status_code == 201
    assert valid.json()["security_symbol"] == "WORLD-ETF"
