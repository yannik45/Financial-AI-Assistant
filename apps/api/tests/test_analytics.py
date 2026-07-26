import math

from financial_ai.analytics import calculate_analytics
from financial_ai.database import SessionLocal
from financial_ai.models import Portfolio
from financial_ai.seed import DEMO_PORTFOLIOS, seed_demo_portfolios


def test_demo_analytics_are_deterministic():
    with SessionLocal() as session:
        seed_demo_portfolios(session)
        portfolio = session.query(Portfolio).filter_by(name="Diversified Global Portfolio").one()
        result = calculate_analytics(portfolio)
    assert result.market_value_eur > 0
    assert result.cost_basis_eur > 0
    assert 0 < result.concentration_hhi < 1
    assert result.max_drawdown_percent <= 0
    assert len(result.value_series) > 40
    assert math.isclose(sum(item.weight for item in result.positions), 1.0, abs_tol=1e-5)


def test_all_five_demo_profiles_exist():
    assert len(DEMO_PORTFOLIOS) == 5
