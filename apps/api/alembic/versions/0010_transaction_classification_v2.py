"""Store transaction classification v2 provenance.

Revision ID: 0010_classification_v2
Revises: 0009_market_price_ohlc
"""

import sqlalchemy as sa
from alembic import op

revision = "0010_classification_v2"
down_revision = "0009_market_price_ohlc"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("transaction_classifications") as batch_op:
        batch_op.add_column(
            sa.Column(
                "input_source",
                sa.String(length=30),
                nullable=False,
                server_default="manual_entry",
            )
        )
        batch_op.add_column(
            sa.Column("alternative_predicted_category", sa.String(length=60), nullable=True)
        )
        batch_op.add_column(
            sa.Column("alternative_model_version", sa.String(length=120), nullable=True)
        )
        batch_op.add_column(sa.Column("model_agreement", sa.Boolean(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("transaction_classifications") as batch_op:
        batch_op.drop_column("model_agreement")
        batch_op.drop_column("alternative_model_version")
        batch_op.drop_column("alternative_predicted_category")
        batch_op.drop_column("input_source")
