from financial_ai.importer import EXPECTED_COLUMNS


def test_health_and_correlation_id(client):
    response = client.get("/health", headers={"X-Correlation-ID": "test-request"})
    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "test-request"


def test_demo_portfolio_can_be_analyzed(client):
    portfolios = client.get("/v1/portfolios")
    assert portfolios.status_code == 200
    assert len(portfolios.json()) == 5
    portfolio_id = portfolios.json()[0]["id"]
    response = client.get(f"/v1/portfolios/{portfolio_id}/analytics")
    assert response.status_code == 200
    assert response.json()["data_version"] == "demo-market-2026.1"


def test_csv_import_is_atomic(client):
    valid = "\n".join(
        [
            ",".join(EXPECTED_COLUMNS),
            "WORLD-ETF,10,104,2024-02-15,Equity ETF,Broad Market,Global,EUR",
        ]
    )
    response = client.post(
        "/v1/portfolios/import",
        data={"name": "Imported"},
        files={"file": ("positions.csv", valid, "text/csv")},
    )
    assert response.status_code == 201
    assert len(client.get("/v1/portfolios").json()) == 6

    invalid = valid + "\nUNKNOWN,1,10,2024-01-01,Equity,Other,Global,EUR"
    response = client.post(
        "/v1/portfolios/import",
        data={"name": "Invalid"},
        files={"file": ("positions.csv", invalid, "text/csv")},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_portfolio_csv"
    assert len(client.get("/v1/portfolios").json()) == 6
