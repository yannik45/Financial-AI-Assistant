from datetime import date
from decimal import Decimal
from uuid import UUID, uuid5

from sqlalchemy import select
from sqlalchemy.orm import Session

from financial_ai.market_data import ASSETS
from financial_ai.models import Account, Portfolio, Position, Transaction

DEMO_PORTFOLIOS: dict[str, list[tuple[str, Decimal]]] = {
    "Diversified Global Portfolio": [
        ("WORLD-ETF", Decimal("55")),
        ("EURO-BOND", Decimal("40")),
        ("GOLD-ETC", Decimal("12")),
        ("EU-HEALTH", Decimal("20")),
    ],
    "Technology Concentration": [
        ("US-TECH-A", Decimal("70")),
        ("US-TECH-B", Decimal("55")),
        ("EU-TECH", Decimal("45")),
    ],
    "Single Position Concentration": [
        ("WORLD-ETF", Decimal("12")),
        ("US-TECH-A", Decimal("150")),
        ("EURO-BOND", Decimal("10")),
    ],
    "Defensive ETF Portfolio": [
        ("EURO-BOND", Decimal("110")),
        ("WORLD-ETF", Decimal("25")),
        ("UK-DIVIDEND", Decimal("30")),
        ("GOLD-ETC", Decimal("8")),
    ],
    "International FX Portfolio": [
        ("US-HEALTH", Decimal("45")),
        ("UK-DIVIDEND", Decimal("65")),
        ("JP-EQUITY", Decimal("1.2")),
        ("WORLD-ETF", Decimal("30")),
    ],
}

DEMO_NAMESPACE = UUID("87b63f0b-4f2f-4c50-98f8-f64057eeea2d")
DEMO_ACCOUNTS = {
    "checking": ("Main Checking", "checking"),
    "savings": ("Emergency Savings", "savings"),
    "brokerage": ("Diversified Global Portfolio Brokerage", "brokerage"),
}

DEMO_TRANSACTIONS = [
    (
        "checking",
        "2026-01-02",
        "Monthly salary",
        "3200.00",
        "salary",
        "Employer Demo GmbH",
        "Income",
    ),
    (
        "checking",
        "2026-01-04",
        "Apartment rent",
        "-1120.00",
        "direct_debit",
        "Demo Property GmbH",
        "Housing",
    ),
    (
        "checking",
        "2026-01-07",
        "Supermarket",
        "-84.35",
        "card_payment",
        "Fresh Market",
        "Groceries",
    ),
    (
        "checking",
        "2026-01-12",
        "Electricity bill",
        "-76.00",
        "direct_debit",
        "Demo Energy",
        "Utilities",
    ),
    (
        "checking",
        "2026-02-02",
        "Monthly salary",
        "3200.00",
        "salary",
        "Employer Demo GmbH",
        "Income",
    ),
    (
        "checking",
        "2026-02-04",
        "Apartment rent",
        "-1120.00",
        "direct_debit",
        "Demo Property GmbH",
        "Housing",
    ),
    ("checking", "2026-02-09", "Train ticket", "-49.90", "card_payment", "Demo Rail", "Transport"),
    ("checking", "2026-02-15", "Restaurant", "-68.40", "card_payment", "Demo Bistro", "Dining"),
    (
        "checking",
        "2026-03-02",
        "Monthly salary",
        "3200.00",
        "salary",
        "Employer Demo GmbH",
        "Income",
    ),
    (
        "checking",
        "2026-03-04",
        "Apartment rent",
        "-1120.00",
        "direct_debit",
        "Demo Property GmbH",
        "Housing",
    ),
    (
        "checking",
        "2026-03-10",
        "Supermarket",
        "-92.18",
        "card_payment",
        "Fresh Market",
        "Groceries",
    ),
    ("checking", "2026-03-18", "Cash withdrawal", "-100.00", "cash_withdrawal", "Demo ATM", "Cash"),
    ("savings", "2026-01-03", "Savings deposit", "500.00", "deposit", "Main Checking", "Savings"),
    ("savings", "2026-03-31", "Quarterly interest", "4.82", "interest", "Demo Bank", "Interest"),
]

DEMO_SECURITY_TRANSACTIONS = [
    (
        "2026-01-08",
        "Buy WORLD-ETF",
        "-1043.50",
        "security_buy",
        "WORLD-ETF",
        "10",
        "104.00",
        "3.50",
        "0.00",
    ),
    (
        "2026-02-20",
        "Buy EU-TECH",
        "-724.90",
        "security_buy",
        "EU-TECH",
        "8",
        "90.00",
        "4.90",
        "0.00",
    ),
    ("2026-03-15", "WORLD-ETF dividend", "18.20", "dividend", None, None, None, "0.00", "4.55"),
    ("2026-03-27", "Brokerage account fee", "-5.00", "fee", None, None, None, "0.00", "0.00"),
]


def seed_demo_portfolios(session: Session) -> None:
    if session.scalar(select(Portfolio.id).where(Portfolio.kind == "demo").limit(1)):
        return
    for portfolio_index, (name, holdings) in enumerate(DEMO_PORTFOLIOS.items()):
        account = Account(
            id=(
                str(uuid5(DEMO_NAMESPACE, "account:brokerage"))
                if portfolio_index == 0
                else str(uuid5(DEMO_NAMESPACE, f"portfolio-account:{name}"))
            ),
            name=f"{name} Brokerage",
            account_type="brokerage",
            currency="EUR",
            kind="demo",
            opening_balance=Decimal("10000.00"),
        )
        portfolio = Portfolio(name=name, kind="demo", account=account)
        for position_index, (symbol, quantity) in enumerate(holdings):
            asset = ASSETS[symbol]
            portfolio.positions.append(
                Position(
                    symbol=symbol,
                    quantity=quantity,
                    purchase_price=Decimal(
                        str(round(asset.start_price * (0.92 + 0.015 * position_index), 2))
                    ),
                    purchase_date=date(2024, 2 + portfolio_index, 15),
                    asset_class=asset.asset_class,
                    sector=asset.sector,
                    region=asset.region,
                    currency=asset.currency,
                )
            )
        session.add(portfolio)
    session.commit()


def seed_demo_accounts(session: Session) -> None:
    checking_id = str(uuid5(DEMO_NAMESPACE, "account:checking"))
    if session.get(Account, checking_id):
        return

    accounts: dict[str, Account] = {}
    for key, (name, account_type) in DEMO_ACCOUNTS.items():
        if key == "brokerage":
            brokerage = session.get(Account, str(uuid5(DEMO_NAMESPACE, "account:brokerage")))
            if brokerage is None:
                raise RuntimeError("Demo portfolio brokerage account was not seeded")
            accounts[key] = brokerage
            continue
        account = Account(
            id=str(uuid5(DEMO_NAMESPACE, f"account:{key}")),
            name=name,
            account_type=account_type,
            currency="EUR",
            kind="demo",
        )
        accounts[key] = account
        session.add(account)

    for index, item in enumerate(DEMO_TRANSACTIONS):
        account_key, booked_at, name, amount, transaction_type, counterparty, category = item
        accounts[account_key].transactions.append(
            Transaction(
                id=str(uuid5(DEMO_NAMESPACE, f"transaction:cash:{index}")),
                booked_at=date.fromisoformat(booked_at),
                name=name,
                amount=Decimal(amount),
                currency="EUR",
                transaction_type=transaction_type,
                counterparty=counterparty,
                category=category,
                source="demo",
            )
        )

    brokerage = accounts["brokerage"]
    for index, item in enumerate(DEMO_SECURITY_TRANSACTIONS):
        booked_at, name, amount, transaction_type, symbol, quantity, unit_price, fees, taxes = item
        brokerage.transactions.append(
            Transaction(
                id=str(uuid5(DEMO_NAMESPACE, f"transaction:security:{index}")),
                booked_at=date.fromisoformat(booked_at),
                name=name,
                amount=Decimal(amount),
                currency="EUR",
                transaction_type=transaction_type,
                counterparty="Demo Broker",
                category="Investments",
                source="demo",
                security_symbol=symbol,
                quantity=Decimal(quantity) if quantity else None,
                unit_price=Decimal(unit_price) if unit_price else None,
                fees=Decimal(fees),
                taxes=Decimal(taxes),
            )
        )

    session.commit()
