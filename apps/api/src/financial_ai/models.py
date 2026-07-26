from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import Date, DateTime, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from financial_ai.database import Base


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(120))
    base_currency: Mapped[str] = mapped_column(String(3), default="EUR")
    kind: Mapped[str] = mapped_column(String(20), default="imported")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
    positions: Mapped[list["Position"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan", lazy="selectin"
    )


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_id: Mapped[str] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"))
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    purchase_price: Mapped[Decimal] = mapped_column(Numeric(20, 6))
    purchase_date: Mapped[date] = mapped_column(Date)
    asset_class: Mapped[str] = mapped_column(String(40))
    sector: Mapped[str] = mapped_column(String(60))
    region: Mapped[str] = mapped_column(String(40))
    currency: Mapped[str] = mapped_column(String(3))
    portfolio: Mapped[Portfolio] = relationship(back_populates="positions")
