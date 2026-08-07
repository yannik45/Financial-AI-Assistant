"""Add open, high, and low values to cached daily market prices.

Revision ID: 0009_market_price_ohlc
Revises: 0008_market_data_mode
"""

import sqlalchemy as sa
from alembic import op

revision = "0009_market_price_ohlc"
down_revision = "0008_market_data_mode"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("market_price_observations") as batch_op:
        batch_op.add_column(sa.Column("open", sa.Numeric(20, 8), nullable=True))
        batch_op.add_column(sa.Column("high", sa.Numeric(20, 8), nullable=True))
        batch_op.add_column(sa.Column("low", sa.Numeric(20, 8), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("market_price_observations") as batch_op:
        batch_op.drop_column("low")
        batch_op.drop_column("high")
        batch_op.drop_column("open")
