from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from financial_ai.market_data import ASSETS
from financial_ai.models import Portfolio, Position


DEMO_PORTFOLIOS: dict[str, list[tuple[str, Decimal]]] = {
    "Diversified Global Portfolio": [("WORLD-ETF", Decimal("55")), ("EURO-BOND", Decimal("40")), ("GOLD-ETC", Decimal("12")), ("EU-HEALTH", Decimal("20"))],
    "Technology Concentration": [("US-TECH-A", Decimal("70")), ("US-TECH-B", Decimal("55")), ("EU-TECH", Decimal("45"))],
    "Single Position Concentration": [("WORLD-ETF", Decimal("12")), ("US-TECH-A", Decimal("150")), ("EURO-BOND", Decimal("10"))],
    "Defensive ETF Portfolio": [("EURO-BOND", Decimal("110")), ("WORLD-ETF", Decimal("25")), ("UK-DIVIDEND", Decimal("30")), ("GOLD-ETC", Decimal("8"))],
    "International FX Portfolio": [("US-HEALTH", Decimal("45")), ("UK-DIVIDEND", Decimal("65")), ("JP-EQUITY", Decimal("1.2")), ("WORLD-ETF", Decimal("30"))],
}


def seed_demo_portfolios(session: Session) -> None:
    if session.scalar(select(Portfolio.id).where(Portfolio.kind == "demo").limit(1)):
        return
    for portfolio_index, (name, holdings) in enumerate(DEMO_PORTFOLIOS.items()):
        portfolio = Portfolio(name=name, kind="demo")
        for position_index, (symbol, quantity) in enumerate(holdings):
            asset = ASSETS[symbol]
            portfolio.positions.append(
                Position(
                    symbol=symbol,
                    quantity=quantity,
                    purchase_price=Decimal(str(round(asset.start_price * (0.92 + 0.015 * position_index), 2))),
                    purchase_date=date(2024, 2 + portfolio_index, 15),
                    asset_class=asset.asset_class,
                    sector=asset.sector,
                    region=asset.region,
                    currency=asset.currency,
                )
            )
        session.add(portfolio)
    session.commit()

