from datetime import date
from decimal import Decimal

import pytest
from financial_ai.risk_score import COMPONENTS, calculate_risk_score, interpolate
from financial_ai.schemas import AllocationItem, AnalyticsResponse


def analytics_fixture(**updates) -> AnalyticsResponse:
    values = {
        "portfolio_id": "portfolio-id",
        "as_of": date(2026, 6, 30),
        "data_version": "test-v1",
        "market_value_eur": Decimal("100000.00"),
        "cost_basis_eur": Decimal("90000.00"),
        "unrealized_pnl_eur": Decimal("10000.00"),
        "unrealized_pnl_percent": 11.11,
        "trailing_return_percent": 8.0,
        "annualized_volatility_percent": 10.0,
        "max_drawdown_percent": -12.0,
        "concentration_hhi": 0.20,
        "largest_position_symbol": "WORLD-ETF",
        "largest_position_weight": 0.30,
        "positions": [],
        "allocations": {
            "asset_class": [
                AllocationItem(label="Equity ETF", value_eur=Decimal("70000"), weight=0.7),
                AllocationItem(label="Bond ETF", value_eur=Decimal("30000"), weight=0.3),
            ],
            "currency": [
                AllocationItem(label="EUR", value_eur=Decimal("75000"), weight=0.75),
                AllocationItem(label="USD", value_eur=Decimal("25000"), weight=0.25),
            ],
            "sector": [
                AllocationItem(label="Technology", value_eur=Decimal("60000"), weight=0.6),
                AllocationItem(label="Broad Market", value_eur=Decimal("40000"), weight=0.4),
            ],
            "region": [],
        },
        "value_series": [],
        "warnings": [],
    }
    values.update(updates)
    return AnalyticsResponse(**values)


def component(score, key: str):
    return next(item for item in score.components if item.key == key)


def test_score_is_explainable_bounded_and_weighted():
    result = calculate_risk_score(analytics_fixture(), base_currency="EUR", cash_value_eur=10000)

    assert 0 <= result.score <= 100
    assert result.level in {"low", "moderate", "elevated", "high"}
    assert sum(item.weight for item in result.components) == pytest.approx(1)
    assert result.score == pytest.approx(sum(item.contribution for item in result.components), 0.1)
    assert len(result.main_drivers) == 3
    assert result.methodology_version == "portfolio-risk-score-v2"
    assert result.diversification.level in {"strong", "adequate", "weak"}
    assert result.liquidity_resilience.level in {"strong", "adequate", "weak"}
    assert "not a regulatory risk class" in result.disclaimer


def test_higher_market_risk_increases_score_without_mixing_in_concentration():
    baseline = calculate_risk_score(analytics_fixture(), base_currency="EUR", cash_value_eur=10000)
    stressed = calculate_risk_score(
        analytics_fixture(
            annualized_volatility_percent=35,
            max_drawdown_percent=-45,
            concentration_hhi=0.65,
            largest_position_weight=0.80,
        ),
        base_currency="EUR",
        cash_value_eur=10000,
    )

    assert stressed.score > baseline.score
    assert component(stressed, "volatility").score > component(baseline, "volatility").score
    assert stressed.diversification.score < baseline.diversification.score


def test_cash_and_currency_are_context_not_market_risk_components():
    no_cash = calculate_risk_score(analytics_fixture(), base_currency="EUR", cash_value_eur=0)
    cash_buffer = calculate_risk_score(
        analytics_fixture(), base_currency="EUR", cash_value_eur=100000
    )
    usd_base = calculate_risk_score(analytics_fixture(), base_currency="USD", cash_value_eur=10000)

    assert cash_buffer.score == no_cash.score
    assert cash_buffer.liquidity_resilience.score > no_cash.liquidity_resilience.score
    assert usd_base.score == no_cash.score
    assert no_cash.diversification.details["foreign_currency_denomination_percent"] == 25
    assert usd_base.diversification.details["foreign_currency_denomination_percent"] == 75


def test_sector_cluster_increases_concentration_even_with_same_positions():
    diversified = analytics_fixture(
        allocations={
            **analytics_fixture().allocations,
            "sector": [
                AllocationItem(label="Technology", value_eur=Decimal("50000"), weight=0.5),
                AllocationItem(label="Healthcare", value_eur=Decimal("50000"), weight=0.5),
            ],
        }
    )
    clustered = analytics_fixture(
        allocations={
            **analytics_fixture().allocations,
            "sector": [AllocationItem(label="Technology", value_eur=Decimal("100000"), weight=1.0)],
        }
    )

    diversified_score = calculate_risk_score(diversified, base_currency="EUR", cash_value_eur=10000)
    clustered_score = calculate_risk_score(clustered, base_currency="EUR", cash_value_eur=10000)

    assert clustered_score.diversification.score < diversified_score.diversification.score


def test_broad_market_fund_receives_limited_look_through_credit():
    single_stock = analytics_fixture(
        concentration_hhi=1,
        largest_position_weight=1,
        allocations={
            **analytics_fixture().allocations,
            "sector": [AllocationItem(label="Technology", value_eur=Decimal("100000"), weight=1)],
        },
    )
    broad_fund = analytics_fixture(
        concentration_hhi=1,
        largest_position_weight=1,
        allocations={
            **analytics_fixture().allocations,
            "sector": [AllocationItem(label="Broad Market", value_eur=Decimal("100000"), weight=1)],
        },
    )

    stock_score = calculate_risk_score(single_stock, base_currency="EUR", cash_value_eur=0)
    fund_score = calculate_risk_score(broad_fund, base_currency="EUR", cash_value_eur=0)

    assert fund_score.score == stock_score.score
    assert fund_score.diversification.score > stock_score.diversification.score


def test_interpolation_clamps_and_component_weights_are_stable():
    assert interpolate(-1, ((0, 0), (10, 100))) == 0
    assert interpolate(5, ((0, 0), (10, 100))) == 50
    assert interpolate(20, ((0, 0), (10, 100))) == 100
    assert sum(item.weight for item in COMPONENTS.values()) == pytest.approx(1)


def test_demo_portfolio_scores_have_plausible_relative_order(client):
    portfolios = client.get("/v1/portfolios").json()
    scores = {
        portfolio["name"]: client.get(f"/v1/portfolios/{portfolio['id']}/analytics").json()[
            "risk_score"
        ]
        for portfolio in portfolios
    }

    assert (
        scores["Single Position Concentration"]["diversification"]["score"]
        < scores["Technology Concentration"]["diversification"]["score"]
        < scores["Diversified Global Portfolio"]["diversification"]["score"]
    )
    assert scores["Defensive ETF Portfolio"]["score"] < scores["Technology Concentration"]["score"]
