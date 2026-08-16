from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace

from financial_ai.analytics import current_holdings
from financial_ai.market_data import BROAD_MARKET_FUND_EXPOSURES


def test_acwi_has_explicit_global_broad_market_exposure():
    exposure = BROAD_MARKET_FUND_EXPOSURES["ACWI"]

    assert exposure.asset_class == "Equity ETF"
    assert exposure.sector == "Broad Market"
    assert exposure.region == "Global"


def test_sector_etfs_are_not_implicitly_treated_as_broad_market_funds():
    assert "QQQ" not in BROAD_MARKET_FUND_EXPOSURES


def test_external_acwi_trade_uses_economic_exposure_in_analytics():
    instrument = SimpleNamespace(
        id="acwi-id",
        symbol="ACWI",
        asset_class="US Equity",
        region="United States",
        currency="USD",
    )
    trade = SimpleNamespace(
        market_instrument_id=instrument.id,
        market_instrument=instrument,
        transaction_type="security_buy",
        quantity=Decimal("2"),
        unit_price=Decimal("100"),
        fees=Decimal("0"),
        booked_at=date(2026, 8, 16),
        created_at=datetime(2026, 8, 16),
    )
    portfolio = SimpleNamespace(
        positions=[],
        account=SimpleNamespace(transactions=[trade]),
    )

    holding = current_holdings(portfolio)[0]

    assert holding.asset_class == "Equity ETF"
    assert holding.sector == "Broad Market"
    assert holding.region == "Global"


def test_existing_acwi_position_uses_economic_exposure_in_analytics():
    position = SimpleNamespace(
        symbol="ACWI",
        quantity=Decimal("8"),
        purchase_price=Decimal("100"),
        purchase_date=date(2026, 8, 16),
        asset_class="US Equity",
        sector="US Equity",
        region="United States",
        currency="USD",
    )
    portfolio = SimpleNamespace(positions=[position], account=None)

    holding = current_holdings(portfolio)[0]

    assert holding.asset_class == "Equity ETF"
    assert holding.sector == "Broad Market"
    assert holding.region == "Global"
