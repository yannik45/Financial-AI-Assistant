"""Store the market-data mode on each portfolio.

Revision ID: 0008_market_data_mode
Revises: 0007_correct_order_dates
"""

import sqlalchemy as sa
from alembic import op

revision = "0008_market_data_mode"
down_revision = "0007_correct_order_dates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("portfolios") as batch_op:
        batch_op.add_column(
            sa.Column(
                "market_data_mode", sa.String(length=20), nullable=False, server_default="demo"
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("portfolios") as batch_op:
        batch_op.drop_column("market_data_mode")
