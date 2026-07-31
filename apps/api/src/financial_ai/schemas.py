from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from financial_ai.ml.transaction_classification import (
    ClassificationMethod,
    ClassificationRoute,
    FeedbackStatus,
    parse_product_category,
)


class PositionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    symbol: str
    quantity: Decimal
    purchase_price: Decimal
    purchase_date: date
    asset_class: str
    sector: str
    region: str
    currency: str


class PortfolioSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    base_currency: str
    kind: str
    created_at: datetime
    position_count: int


class PortfolioRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    base_currency: str
    kind: str
    created_at: datetime
    positions: list[PositionRead]


class CatalogAsset(BaseModel):
    symbol: str
    name: str
    currency: str
    asset_class: str
    sector: str
    region: str


class AllocationItem(BaseModel):
    label: str
    value_eur: Decimal
    weight: float


class PositionAnalytics(BaseModel):
    symbol: str
    market_value_eur: Decimal
    cost_basis_eur: Decimal
    pnl_eur: Decimal
    weight: float


class SeriesPoint(BaseModel):
    date: date
    value_eur: Decimal


class AnalyticsResponse(BaseModel):
    portfolio_id: str
    as_of: date
    data_version: str
    market_value_eur: Decimal
    cost_basis_eur: Decimal
    unrealized_pnl_eur: Decimal
    unrealized_pnl_percent: float
    trailing_return_percent: float
    annualized_volatility_percent: float
    max_drawdown_percent: float
    concentration_hhi: float
    largest_position_symbol: str
    largest_position_weight: float
    positions: list[PositionAnalytics]
    allocations: dict[str, list[AllocationItem]]
    value_series: list[SeriesPoint]
    warnings: list[str] = Field(default_factory=list)


class ApiError(BaseModel):
    code: str
    message: str
    details: list[dict[str, object]] = Field(default_factory=list)


class AccountType(StrEnum):
    CHECKING = "checking"
    SAVINGS = "savings"
    BROKERAGE = "brokerage"


class TransactionType(StrEnum):
    UNSPECIFIED = "unspecified"
    CARD_PAYMENT = "card_payment"
    TRANSFER = "transfer"
    DIRECT_DEBIT = "direct_debit"
    CASH_WITHDRAWAL = "cash_withdrawal"
    SALARY = "salary"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    SECURITY_BUY = "security_buy"
    SECURITY_SELL = "security_sell"
    DIVIDEND = "dividend"
    INTEREST = "interest"
    FEE = "fee"
    TAX = "tax"


class AccountRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    account_type: AccountType
    currency: str
    kind: str
    created_at: datetime
    transaction_count: int | None = None


class TransactionClassificationRecordRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    predicted_category: str | None
    final_category: str | None
    route: ClassificationRoute
    classification_method: ClassificationMethod
    confidence: float | None
    needs_review: bool
    feedback_status: FeedbackStatus
    reason: str
    taxonomy_version: str
    model_version: str | None
    created_at: datetime


class TransactionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    account_id: str
    booked_at: date
    name: str
    amount: Decimal
    currency: str
    transaction_type: TransactionType
    counterparty: str | None
    category: str | None
    notes: str | None
    source: str
    security_symbol: str | None
    quantity: Decimal | None
    unit_price: Decimal | None
    fees: Decimal
    taxes: Decimal
    created_at: datetime
    classifications: list[TransactionClassificationRecordRead] = Field(default_factory=list)


class TransactionCreate(BaseModel):
    account_id: str
    booked_at: date
    name: str = Field(min_length=1, max_length=160)
    amount: Decimal = Field(max_digits=20, decimal_places=2)
    currency: str = Field(default="EUR", min_length=3, max_length=3)
    transaction_type: TransactionType = TransactionType.UNSPECIFIED
    counterparty: str | None = Field(default=None, max_length=160)
    category: str | None = Field(default=None, max_length=60)
    category_confirmed: bool = False
    notes: str | None = Field(default=None, max_length=500)
    security_symbol: str | None = Field(default=None, max_length=24)
    quantity: Decimal | None = Field(default=None, gt=0, max_digits=20, decimal_places=8)
    unit_price: Decimal | None = Field(default=None, gt=0, max_digits=20, decimal_places=6)
    fees: Decimal = Field(default=Decimal("0.00"), ge=0, max_digits=20, decimal_places=2)
    taxes: Decimal = Field(default=Decimal("0.00"), ge=0, max_digits=20, decimal_places=2)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Transaction name must not be empty")
        return normalized

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return parse_product_category(value)

    @model_validator(mode="after")
    def validate_security_fields(self) -> "TransactionCreate":
        if self.amount == 0:
            raise ValueError("Transaction amount must not be zero")
        security_types = {TransactionType.SECURITY_BUY, TransactionType.SECURITY_SELL}
        if self.transaction_type in security_types:
            missing = [
                field
                for field in ("security_symbol", "quantity", "unit_price")
                if getattr(self, field) is None
            ]
            if missing:
                raise ValueError(
                    "Security transactions require security_symbol, quantity, and unit_price"
                )
        elif any(
            value is not None for value in (self.security_symbol, self.quantity, self.unit_price)
        ):
            raise ValueError(
                "Security fields are only allowed for security buy or sell transactions"
            )
        return self


class TransactionPage(BaseModel):
    items: list[TransactionRead]
    total: int
    limit: int
    offset: int


class TransactionClassificationRequest(BaseModel):
    description: str = Field(min_length=1, max_length=320)
    amount: Decimal = Field(max_digits=20, decimal_places=2)
    counterparty: str | None = Field(default=None, max_length=160)


class TransactionClassificationResponse(BaseModel):
    category: str | None
    route: ClassificationRoute
    classification_method: ClassificationMethod
    confidence: float | None = Field(default=None, ge=0, le=1)
    needs_review: bool
    reason: str
    taxonomy_version: str
    model_version: str | None
