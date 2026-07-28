"""Add accounts and transactions.

Revision ID: 0002_transaction_foundation
Revises: 0001_initial
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_transaction_foundation"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "accounts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("account_type", sa.String(20), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_accounts_account_type", "accounts", ["account_type"])

    op.create_table(
        "transactions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "account_id",
            sa.String(36),
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("booked_at", sa.Date(), nullable=False),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("amount", sa.Numeric(20, 2), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("transaction_type", sa.String(30), nullable=False),
        sa.Column("counterparty", sa.String(160), nullable=True),
        sa.Column("category", sa.String(60), nullable=True),
        sa.Column("notes", sa.String(500), nullable=True),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("security_symbol", sa.String(24), nullable=True),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=True),
        sa.Column("unit_price", sa.Numeric(20, 6), nullable=True),
        sa.Column("fees", sa.Numeric(20, 2), nullable=False),
        sa.Column("taxes", sa.Numeric(20, 2), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_transactions_account_id", "transactions", ["account_id"])
    op.create_index("ix_transactions_booked_at", "transactions", ["booked_at"])
    op.create_index("ix_transactions_category", "transactions", ["category"])
    op.create_index("ix_transactions_transaction_type", "transactions", ["transaction_type"])
    op.create_index(
        "ix_transactions_account_booked_at", "transactions", ["account_id", "booked_at"]
    )
    op.create_index(
        "ix_transactions_type_booked_at", "transactions", ["transaction_type", "booked_at"]
    )


def downgrade() -> None:
    op.drop_table("transactions")
    op.drop_table("accounts")
