"""Add paper portfolios and immutable simulated trades.

Revision ID: 0005_paper_trading
Revises: 0004_market_data_foundation
"""

import sqlalchemy as sa
from alembic import op

revision = "0005_paper_trading"
down_revision = "0004_market_data_foundation"
branch_labels = None
depends_on = None


def upgrade() -> None:
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
        sa.Column(
            "portfolio_id",
            sa.String(36),
            sa.ForeignKey("paper_portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "instrument_id",
            sa.String(36),
            sa.ForeignKey("market_instruments.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("client_order_id", sa.String(64), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("unit_price", sa.Numeric(20, 8), nullable=False),
        sa.Column("fees", sa.Numeric(20, 2), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("price_observed_on", sa.Date(), nullable=False),
        sa.Column("price_source", sa.String(30), nullable=False),
        sa.Column("executed_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "portfolio_id", "client_order_id", name="uq_paper_trade_portfolio_client_order"
        ),
    )
    op.create_index("ix_paper_trades_portfolio_id", "paper_trades", ["portfolio_id"])
    op.create_index("ix_paper_trades_instrument_id", "paper_trades", ["instrument_id"])
    op.create_index(
        "ix_paper_trades_portfolio_executed",
        "paper_trades",
        ["portfolio_id", "executed_at"],
    )


def downgrade() -> None:
    op.drop_table("paper_trades")
    op.drop_table("paper_portfolios")
