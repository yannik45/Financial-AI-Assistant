from financial_ai.main import app
from financial_ai.ml.category_artifact import ModelArtifactError
from financial_ai.ml.category_service import (
    TransactionClassification,
    get_transaction_classifier,
)
from financial_ai.ml.transaction_classification import (
    ClassificationMethod,
    ClassificationRoute,
)


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

    inflows = client.get("/v1/transactions", params={"cash_flow": "inflow"})
    assert inflows.status_code == 200
    assert inflows.json()["items"]
    assert all(float(item["amount"]) > 0 for item in inflows.json()["items"])


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
    assert transaction["category"] == "dining"
    assert len(transaction["classifications"]) == 1
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


def test_classification_endpoint_handles_deterministic_and_review_routes(client):
    salary = client.post(
        "/v1/transactions/classify",
        json={
            "description": "Monthly salary Demo GmbH",
            "amount": "2500.00",
        },
    )
    assert salary.status_code == 200
    assert salary.json() == {
        "category": "income",
        "route": "text_rule",
        "classification_method": "keyword_rule",
        "confidence": None,
        "needs_review": False,
        "reason": "Category matched a reviewable text rule in the experimental baseline.",
        "taxonomy_version": "transaction-categories-v1",
        "model_version": None,
    }

    transfer = client.post(
        "/v1/transactions/classify",
        json={"description": "Unknown incoming reference", "amount": "100.00"},
    )
    assert transfer.status_code == 200
    assert transfer.json()["category"] is None
    assert transfer.json()["route"] == "needs_review"
    assert transfer.json()["needs_review"] is True


def test_classification_endpoint_returns_ml_provenance(client):
    class StubClassifier:
        def classify(self, **_):
            return TransactionClassification(
                category="groceries",
                route=ClassificationRoute.EXPENSE_MODEL,
                method=ClassificationMethod.ML,
                confidence=0.81,
                needs_review=False,
                reason="Expense category predicted by the versioned model artifact.",
                taxonomy_version="transaction-categories-v1",
                model_version="test-model-v1",
            )

    app.dependency_overrides[get_transaction_classifier] = lambda: StubClassifier()
    try:
        response = client.post(
            "/v1/transactions/classify",
            json={
                "description": "Demo supermarket purchase",
                "amount": "-20.00",
                "counterparty": "Demo Market",
            },
        )
    finally:
        app.dependency_overrides.pop(get_transaction_classifier, None)

    assert response.status_code == 200
    assert response.json()["category"] == "groceries"
    assert response.json()["classification_method"] == "ml"
    assert response.json()["confidence"] == 0.81
    assert response.json()["model_version"] == "test-model-v1"


def test_classification_endpoint_reports_unavailable_model(client):
    class MissingModelClassifier:
        def classify(self, **_):
            raise ModelArtifactError("Category model artifact is unavailable")

    app.dependency_overrides[get_transaction_classifier] = lambda: MissingModelClassifier()
    try:
        response = client.post(
            "/v1/transactions/classify",
            json={
                "description": "Demo utility charge",
                "amount": "-80.00",
            },
        )
    finally:
        app.dependency_overrides.pop(get_transaction_classifier, None)

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "category_model_unavailable"


def test_transaction_creation_persists_corrected_model_feedback(client):
    class GroceryClassifier:
        def classify(self, **_):
            return TransactionClassification(
                category="groceries",
                route=ClassificationRoute.EXPENSE_MODEL,
                method=ClassificationMethod.ML,
                confidence=0.88,
                needs_review=False,
                reason="Expense category predicted by the versioned model artifact.",
                taxonomy_version="transaction-categories-v1",
                model_version="test-model-v1",
            )

    checking = next(
        account
        for account in client.get("/v1/accounts").json()
        if account["account_type"] == "checking"
    )
    app.dependency_overrides[get_transaction_classifier] = lambda: GroceryClassifier()
    try:
        response = client.post(
            "/v1/transactions",
            json={
                "account_id": checking["id"],
                "booked_at": "2026-04-05",
                "name": "Demo mixed merchant",
                "amount": "-22.00",
                "currency": "EUR",
                "transaction_type": "card_payment",
                "category": "Dining",
                "category_confirmed": True,
            },
        )
    finally:
        app.dependency_overrides.pop(get_transaction_classifier, None)

    assert response.status_code == 201
    transaction = response.json()
    assert transaction["category"] == "dining"
    feedback = transaction["classifications"][0]
    assert feedback["predicted_category"] == "groceries"
    assert feedback["final_category"] == "dining"
    assert feedback["feedback_status"] == "corrected"
    assert feedback["model_version"] == "test-model-v1"


def test_matching_category_tracks_explicit_confirmation(client):
    checking = next(
        account
        for account in client.get("/v1/accounts").json()
        if account["account_type"] == "checking"
    )
    response = client.post(
        "/v1/transactions",
        json={
            "account_id": checking["id"],
            "booked_at": "2026-04-05",
            "name": "House Payment",
            "amount": "-950.00",
            "category": "housing",
            "category_confirmed": True,
        },
    )

    assert response.status_code == 201
    assert response.json()["classifications"][0]["feedback_status"] == "accepted_explicit"


def test_transaction_creation_rejects_unknown_category(client):
    checking = next(
        account
        for account in client.get("/v1/accounts").json()
        if account["account_type"] == "checking"
    )
    response = client.post(
        "/v1/transactions",
        json={
            "account_id": checking["id"],
            "booked_at": "2026-04-05",
            "name": "Demo merchant",
            "amount": "-22.00",
            "transaction_type": "card_payment",
            "category": "invented-category",
        },
    )
    assert response.status_code == 422


def test_transaction_text_not_transaction_type_drives_saved_suggestion(client):
    checking = next(
        account
        for account in client.get("/v1/accounts").json()
        if account["account_type"] == "checking"
    )
    response = client.post(
        "/v1/transactions",
        json={
            "account_id": checking["id"],
            "booked_at": "2026-04-06",
            "name": "House Payment",
            "amount": "-900.00",
            "transaction_type": "salary",
        },
    )
    assert response.status_code == 201
    transaction = response.json()
    assert transaction["transaction_type"] == "salary"
    feedback = transaction["classifications"][0]
    assert feedback["predicted_category"] == "housing"
    assert feedback["route"] == "text_rule"
    assert feedback["classification_method"] == "keyword_rule"
