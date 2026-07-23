"""Initial portfolio schema."""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("portfolios", sa.Column("id", sa.String(36), primary_key=True), sa.Column("name", sa.String(120), nullable=False), sa.Column("base_currency", sa.String(3), nullable=False), sa.Column("kind", sa.String(20), nullable=False), sa.Column("created_at", sa.DateTime(), nullable=False))
    op.create_table("positions", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("portfolio_id", sa.String(36), sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False), sa.Column("symbol", sa.String(24), nullable=False), sa.Column("quantity", sa.Numeric(20, 8), nullable=False), sa.Column("purchase_price", sa.Numeric(20, 6), nullable=False), sa.Column("purchase_date", sa.Date(), nullable=False), sa.Column("asset_class", sa.String(40), nullable=False), sa.Column("sector", sa.String(60), nullable=False), sa.Column("region", sa.String(40), nullable=False), sa.Column("currency", sa.String(3), nullable=False))
    op.create_index("ix_positions_symbol", "positions", ["symbol"])


def downgrade() -> None:
    op.drop_table("positions")
    op.drop_table("portfolios")

