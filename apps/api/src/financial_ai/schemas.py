from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


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
