"""Separate market-order booking dates from quote observation dates.

Revision ID: 0007_correct_order_dates
Revises: 0006_unified_portfolio_ledger
"""

import sqlalchemy as sa
from alembic import op

revision = "0007_correct_order_dates"
down_revision = "0006_unified_portfolio_ledger"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    connection.execute(
        sa.text(
            "UPDATE transactions "
            "SET booked_at = date(created_at) "
            "WHERE source = 'market_order' "
            "AND price_observed_on IS NOT NULL "
            "AND booked_at = price_observed_on"
        )
    )


def downgrade() -> None:
    connection = op.get_bind()
    connection.execute(
        sa.text(
            "UPDATE transactions "
            "SET booked_at = price_observed_on "
            "WHERE source = 'market_order' "
            "AND price_observed_on IS NOT NULL"
        )
    )
