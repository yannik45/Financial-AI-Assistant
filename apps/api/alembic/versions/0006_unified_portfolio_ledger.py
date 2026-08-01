"""Unify portfolios, brokerage cash, and security transactions.

Revision ID: 0006_unified_portfolio_ledger
Revises: 0005_paper_trading
"""

from uuid import uuid4

import sqlalchemy as sa
from alembic import op

revision = "0006_unified_portfolio_ledger"
down_revision = "0005_paper_trading"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "accounts",
        sa.Column("opening_balance", sa.Numeric(20, 2), nullable=False, server_default="0.00"),
    )
    with op.batch_alter_table("portfolios") as batch:
        batch.add_column(sa.Column("account_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_portfolios_account_id", "accounts", ["account_id"], ["id"], ondelete="RESTRICT"
        )
        batch.create_unique_constraint("uq_portfolios_account_id", ["account_id"])

    with op.batch_alter_table("transactions") as batch:
        batch.add_column(sa.Column("market_instrument_id", sa.String(36), nullable=True))
        batch.add_column(sa.Column("client_order_id", sa.String(64), nullable=True))
        batch.add_column(sa.Column("price_observed_on", sa.Date(), nullable=True))
        batch.add_column(sa.Column("price_source", sa.String(30), nullable=True))
        batch.create_foreign_key(
            "fk_transactions_market_instrument_id",
            "market_instruments",
            ["market_instrument_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch.create_unique_constraint(
            "uq_transaction_account_order", ["account_id", "client_order_id"]
        )
        batch.create_index(
            "ix_transactions_market_instrument_id", ["market_instrument_id"], unique=False
        )

    connection = op.get_bind()
    portfolios = connection.execute(
        sa.text("SELECT id, name, base_currency, kind FROM portfolios ORDER BY created_at, name")
    ).mappings()
    demo_brokerage = connection.execute(
        sa.text(
            "SELECT id FROM accounts WHERE account_type = 'brokerage' "
            "AND kind = 'demo' "
            "ORDER BY created_at LIMIT 1"
        )
    ).scalar_one_or_none()
    used_demo_brokerage = False
    for portfolio in portfolios:
        if portfolio["kind"] == "demo" and demo_brokerage and not used_demo_brokerage:
            account_id = demo_brokerage
            used_demo_brokerage = True
            connection.execute(
                sa.text("UPDATE accounts SET opening_balance = 10000.00 WHERE id = :id"),
                {"id": account_id},
            )
        else:
            account_id = str(uuid4())
            connection.execute(
                sa.text(
                    "INSERT INTO accounts "
                    "(id, name, account_type, currency, kind, opening_balance, created_at) "
                    "VALUES (:id, :name, 'brokerage', :currency, :kind, "
                    "10000.00, CURRENT_TIMESTAMP)"
                ),
                {
                    "id": account_id,
                    "name": f"{portfolio['name']} Brokerage",
                    "currency": portfolio["base_currency"],
                    "kind": portfolio["kind"],
                },
            )
        connection.execute(
            sa.text("UPDATE portfolios SET account_id = :account_id WHERE id = :portfolio_id"),
            {"account_id": account_id, "portfolio_id": portfolio["id"]},
        )

    paper_portfolios = connection.execute(sa.text("SELECT * FROM paper_portfolios")).mappings()
    for paper in paper_portfolios:
        portfolio_id = str(uuid4())
        account_id = str(uuid4())
        connection.execute(
            sa.text(
                "INSERT INTO accounts "
                "(id, name, account_type, currency, kind, opening_balance, created_at) "
                "VALUES (:id, :name, 'brokerage', :currency, 'manual', :cash, :created_at)"
            ),
            {
                "id": account_id,
                "name": f"{paper['name']} Brokerage",
                "currency": paper["base_currency"],
                "cash": paper["starting_cash"],
                "created_at": paper["created_at"],
            },
        )
        connection.execute(
            sa.text(
                "INSERT INTO portfolios "
                "(id, name, base_currency, kind, account_id, created_at) "
                "VALUES (:id, :name, :currency, 'manual', :account_id, :created_at)"
            ),
            {
                "id": portfolio_id,
                "name": paper["name"],
                "currency": paper["base_currency"],
                "account_id": account_id,
                "created_at": paper["created_at"],
            },
        )
        trades = connection.execute(
            sa.text("SELECT * FROM paper_trades WHERE portfolio_id = :id"),
            {"id": paper["id"]},
        ).mappings()
        for trade in trades:
            signed_amount = trade["quantity"] * trade["unit_price"]
            if trade["side"] == "buy":
                signed_amount = -signed_amount
            connection.execute(
                sa.text(
                    "INSERT INTO transactions "
                    "(id, account_id, booked_at, name, amount, currency, transaction_type, "
                    "category, source, market_instrument_id, client_order_id, security_symbol, "
                    "quantity, unit_price, fees, taxes, price_observed_on, price_source, "
                    "created_at) "
                    "SELECT :id, :account_id, :booked_at, :name, :amount, :currency, :type, "
                    "'Investments', 'market_order', :instrument_id, :client_order_id, symbol, "
                    ":quantity, :unit_price, :fees, 0.00, :booked_at, :price_source, :created_at "
                    "FROM market_instruments WHERE id = :instrument_id"
                ),
                {
                    "id": trade["id"],
                    "account_id": account_id,
                    "booked_at": trade["price_observed_on"],
                    "name": f"{trade['side'].title()} market instrument",
                    "amount": signed_amount,
                    "currency": trade["currency"],
                    "type": f"security_{trade['side']}",
                    "instrument_id": trade["instrument_id"],
                    "client_order_id": trade["client_order_id"],
                    "quantity": trade["quantity"],
                    "unit_price": trade["unit_price"],
                    "fees": trade["fees"],
                    "price_source": trade["price_source"],
                    "created_at": trade["executed_at"],
                },
            )

    op.drop_table("paper_trades")
    op.drop_table("paper_portfolios")


def downgrade() -> None:
    op.create_table(
        "paper_portfolios",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("base_currency", sa.String(3), nullable=False),
        sa.Column("starting_cash", sa.Numeric(20, 2), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "paper_trades",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("portfolio_id", sa.String(36), nullable=False),
        sa.Column("instrument_id", sa.String(36), nullable=False),
        sa.Column("client_order_id", sa.String(64), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("unit_price", sa.Numeric(20, 8), nullable=False),
        sa.Column("fees", sa.Numeric(20, 2), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("price_observed_on", sa.Date(), nullable=False),
        sa.Column("price_source", sa.String(30), nullable=False),
        sa.Column("executed_at", sa.DateTime(), nullable=False),
    )
    with op.batch_alter_table("transactions") as batch:
        batch.drop_index("ix_transactions_market_instrument_id")
        batch.drop_constraint("uq_transaction_account_order", type_="unique")
        batch.drop_constraint("fk_transactions_market_instrument_id", type_="foreignkey")
        batch.drop_column("price_source")
        batch.drop_column("price_observed_on")
        batch.drop_column("client_order_id")
        batch.drop_column("market_instrument_id")
    with op.batch_alter_table("portfolios") as batch:
        batch.drop_constraint("uq_portfolios_account_id", type_="unique")
        batch.drop_constraint("fk_portfolios_account_id", type_="foreignkey")
        batch.drop_column("account_id")
    op.drop_column("accounts", "opening_balance")
