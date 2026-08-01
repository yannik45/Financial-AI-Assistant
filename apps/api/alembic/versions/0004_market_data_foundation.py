"""Add cached market instruments and daily price observations.

Revision ID: 0004_market_data_foundation
Revises: 0003_classification_feedback
"""

import sqlalchemy as sa
from alembic import op

revision = "0004_market_data_foundation"
down_revision = "0003_classification_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "market_instruments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("provider", sa.String(30), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("exchange", sa.String(80), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("asset_class", sa.String(40), nullable=False),
        sa.Column("region", sa.String(60), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("discovered_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "provider",
            "exchange",
            "symbol",
            name="uq_market_instrument_provider_exchange_symbol",
        ),
    )
    op.create_index("ix_market_instruments_provider", "market_instruments", ["provider"])
    op.create_index("ix_market_instruments_symbol", "market_instruments", ["symbol"])
    op.create_table(
        "market_price_observations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "instrument_id",
            sa.String(36),
            sa.ForeignKey("market_instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("observed_on", sa.Date(), nullable=False),
        sa.Column("close", sa.Numeric(20, 8), nullable=False),
        sa.Column("adjusted_close", sa.Numeric(20, 8), nullable=True),
        sa.Column("volume", sa.Numeric(24, 4), nullable=True),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("instrument_id", "observed_on", name="uq_market_price_instrument_date"),
    )
    op.create_index(
        "ix_market_price_observations_instrument_id",
        "market_price_observations",
        ["instrument_id"],
    )
    op.create_index(
        "ix_market_prices_instrument_observed",
        "market_price_observations",
        ["instrument_id", "observed_on"],
    )


def downgrade() -> None:
    op.drop_table("market_price_observations")
    op.drop_table("market_instruments")
