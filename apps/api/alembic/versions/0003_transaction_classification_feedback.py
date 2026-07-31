"""Add transaction classification feedback records.

Revision ID: 0003_classification_feedback
Revises: 0002_transaction_foundation
"""

import sqlalchemy as sa
from alembic import op

revision = "0003_classification_feedback"
down_revision = "0002_transaction_foundation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "transaction_classifications",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "transaction_id",
            sa.String(36),
            sa.ForeignKey("transactions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("predicted_category", sa.String(60), nullable=True),
        sa.Column("final_category", sa.String(60), nullable=True),
        sa.Column("route", sa.String(30), nullable=False),
        sa.Column("classification_method", sa.String(20), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("needs_review", sa.Boolean(), nullable=False),
        sa.Column("feedback_status", sa.String(20), nullable=False),
        sa.Column("reason", sa.String(240), nullable=False),
        sa.Column("taxonomy_version", sa.String(80), nullable=False),
        sa.Column("model_version", sa.String(120), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_transaction_classifications_transaction_id",
        "transaction_classifications",
        ["transaction_id"],
    )
    op.create_index(
        "ix_transaction_classifications_feedback_status",
        "transaction_classifications",
        ["feedback_status"],
    )
    op.create_index(
        "ix_transaction_classifications_transaction_created",
        "transaction_classifications",
        ["transaction_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("transaction_classifications")
