from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from financial_ai.database import Base


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(120))
    base_currency: Mapped[str] = mapped_column(String(3), default="EUR")
    kind: Mapped[str] = mapped_column(String(20), default="imported")
    market_data_mode: Mapped[str] = mapped_column(String(20), default="demo")
    account_id: Mapped[str | None] = mapped_column(
        ForeignKey("accounts.id", ondelete="RESTRICT"), unique=True, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
    positions: Mapped[list["Position"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan", lazy="selectin"
    )
    account: Mapped["Account | None"] = relationship(back_populates="portfolio", lazy="joined")


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


class MarketInstrument(Base):
    __tablename__ = "market_instruments"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "exchange",
            "symbol",
            name="uq_market_instrument_provider_exchange_symbol",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    provider: Mapped[str] = mapped_column(String(30), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(200))
    exchange: Mapped[str] = mapped_column(String(80), default="")
    currency: Mapped[str] = mapped_column(String(3))
    asset_class: Mapped[str] = mapped_column(String(40))
    region: Mapped[str | None] = mapped_column(String(60), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
    prices: Mapped[list["MarketPriceObservation"]] = relationship(
        back_populates="instrument", cascade="all, delete-orphan", lazy="selectin"
    )


class MarketPriceObservation(Base):
    __tablename__ = "market_price_observations"
    __table_args__ = (
        UniqueConstraint("instrument_id", "observed_on", name="uq_market_price_instrument_date"),
        Index("ix_market_prices_instrument_observed", "instrument_id", "observed_on"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    instrument_id: Mapped[str] = mapped_column(
        ForeignKey("market_instruments.id", ondelete="CASCADE"), index=True
    )
    observed_on: Mapped[date] = mapped_column(Date)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    open: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    high: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    low: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    adjusted_close: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    volume: Mapped[Decimal | None] = mapped_column(Numeric(24, 4), nullable=True)
    source: Mapped[str] = mapped_column(String(30))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime)
    instrument: Mapped[MarketInstrument] = relationship(back_populates="prices")


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(120))
    account_type: Mapped[str] = mapped_column(String(20), index=True)
    currency: Mapped[str] = mapped_column(String(3), default="EUR")
    kind: Mapped[str] = mapped_column(String(20), default="manual")
    opening_balance: Mapped[Decimal] = mapped_column(Numeric(20, 2), default=Decimal("0.00"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="account", cascade="all, delete-orphan", lazy="selectin"
    )
    portfolio: Mapped["Portfolio | None"] = relationship(back_populates="account")


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        UniqueConstraint("account_id", "client_order_id", name="uq_transaction_account_order"),
        Index("ix_transactions_account_booked_at", "account_id", "booked_at"),
        Index("ix_transactions_type_booked_at", "transaction_type", "booked_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    account_id: Mapped[str] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    booked_at: Mapped[date] = mapped_column(Date, index=True)
    name: Mapped[str] = mapped_column(String(160))
    amount: Mapped[Decimal] = mapped_column(Numeric(20, 2))
    currency: Mapped[str] = mapped_column(String(3))
    transaction_type: Mapped[str] = mapped_column(String(30), index=True)
    counterparty: Mapped[str | None] = mapped_column(String(160), nullable=True)
    category: Mapped[str | None] = mapped_column(String(60), index=True, nullable=True)
    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="manual")
    market_instrument_id: Mapped[str | None] = mapped_column(
        ForeignKey("market_instruments.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    client_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    security_symbol: Mapped[str | None] = mapped_column(String(24), nullable=True)
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    unit_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 6), nullable=True)
    fees: Mapped[Decimal] = mapped_column(Numeric(20, 2), default=Decimal("0.00"))
    taxes: Mapped[Decimal] = mapped_column(Numeric(20, 2), default=Decimal("0.00"))
    price_observed_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    price_source: Mapped[str | None] = mapped_column(String(30), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
    account: Mapped[Account] = relationship(back_populates="transactions")
    market_instrument: Mapped[MarketInstrument | None] = relationship(lazy="joined")
    classifications: Mapped[list["TransactionClassificationRecord"]] = relationship(
        back_populates="transaction",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="TransactionClassificationRecord.created_at",
    )


class TransactionClassificationRecord(Base):
    __tablename__ = "transaction_classifications"
    __table_args__ = (
        Index("ix_transaction_classifications_transaction_created", "transaction_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    transaction_id: Mapped[str] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"), index=True
    )
    predicted_category: Mapped[str | None] = mapped_column(String(60), nullable=True)
    final_category: Mapped[str | None] = mapped_column(String(60), nullable=True)
    route: Mapped[str] = mapped_column(String(30))
    classification_method: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean)
    feedback_status: Mapped[str] = mapped_column(String(20), index=True)
    reason: Mapped[str] = mapped_column(String(240))
    taxonomy_version: Mapped[str] = mapped_column(String(80))
    model_version: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
    transaction: Mapped[Transaction] = relationship(back_populates="classifications")
