from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from financial_ai.market_data import DATA_VERSION, DemoMarketDataProvider, market_data_provider
from financial_ai.models import Portfolio
from financial_ai.schemas import (
    AllocationItem,
    AnalyticsResponse,
    PositionAnalytics,
    SeriesPoint,
)

MONEY = Decimal("0.01")


class AnalyticsError(ValueError):
    pass


def money(value: float) -> Decimal:
    return Decimal(str(value)).quantize(MONEY, rounding=ROUND_HALF_UP)


def calculate_analytics(
    portfolio: Portfolio, provider: DemoMarketDataProvider = market_data_provider
) -> AnalyticsResponse:
    if not portfolio.positions:
        raise AnalyticsError("Portfolio has no positions")

    symbols = sorted({position.symbol for position in portfolio.positions})
    prices = provider.prices(symbols).tail(252)
    if len(prices.dropna()) < 60:
        raise AnalyticsError("At least 60 common price observations are required")

    eur_prices = prices.copy()
    for symbol in symbols:
        currency = next(p.currency for p in portfolio.positions if p.symbol == symbol)
        fx = provider.eur_per_currency(currency).reindex(prices.index).ffill()
        eur_prices[symbol] = prices[symbol] * fx

    quantities = pd.Series(
        {
            symbol: sum(float(p.quantity) for p in portfolio.positions if p.symbol == symbol)
            for symbol in symbols
        }
    )
    value_series = eur_prices.mul(quantities, axis=1)
    portfolio_series = value_series.sum(axis=1)
    current_by_symbol = value_series.iloc[-1]
    market_value = float(current_by_symbol.sum())

    cost_by_symbol: dict[str, float] = defaultdict(float)
    for position in portfolio.positions:
        purchase_fx = provider.fx_on_or_before(position.currency, position.purchase_date)
        cost_by_symbol[position.symbol] += (
            float(position.quantity) * float(position.purchase_price) * purchase_fx
        )
    cost_basis = sum(cost_by_symbol.values())
    pnl = market_value - cost_basis
    weights = current_by_symbol / market_value

    daily_returns = portfolio_series.pct_change().dropna()
    trailing_return = portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1
    annualized_volatility = float(daily_returns.std(ddof=1) * np.sqrt(252))
    drawdown = portfolio_series / portfolio_series.cummax() - 1
    max_drawdown = float(drawdown.min())
    hhi = float((weights**2).sum())
    largest_symbol = str(weights.idxmax())

    position_results = [
        PositionAnalytics(
            symbol=symbol,
            market_value_eur=money(float(current_by_symbol[symbol])),
            cost_basis_eur=money(cost_by_symbol[symbol]),
            pnl_eur=money(float(current_by_symbol[symbol]) - cost_by_symbol[symbol]),
            weight=round(float(weights[symbol]), 6),
        )
        for symbol in sorted(symbols, key=lambda item: weights[item], reverse=True)
    ]

    dimensions = {
        "asset_class": lambda p: p.asset_class,
        "sector": lambda p: p.sector,
        "region": lambda p: p.region,
        "currency": lambda p: p.currency,
    }
    allocations: dict[str, list[AllocationItem]] = {}
    for dimension, getter in dimensions.items():
        grouped: dict[str, float] = defaultdict(float)
        for position in portfolio.positions:
            symbol_total_quantity = quantities[position.symbol]
            position_share = float(position.quantity) / symbol_total_quantity
            grouped[getter(position)] += float(current_by_symbol[position.symbol]) * position_share
        allocations[dimension] = [
            AllocationItem(
                label=label, value_eur=money(value), weight=round(value / market_value, 6)
            )
            for label, value in sorted(grouped.items(), key=lambda item: item[1], reverse=True)
        ]

    sampled_series = portfolio_series.iloc[::5]
    if sampled_series.index[-1] != portfolio_series.index[-1]:
        sampled_series = pd.concat([sampled_series, portfolio_series.iloc[[-1]]])

    return AnalyticsResponse(
        portfolio_id=portfolio.id,
        as_of=portfolio_series.index[-1].date(),
        data_version=DATA_VERSION,
        market_value_eur=money(market_value),
        cost_basis_eur=money(cost_basis),
        unrealized_pnl_eur=money(pnl),
        unrealized_pnl_percent=round((pnl / cost_basis) * 100, 4) if cost_basis else 0,
        trailing_return_percent=round(float(trailing_return) * 100, 4),
        annualized_volatility_percent=round(annualized_volatility * 100, 4),
        max_drawdown_percent=round(max_drawdown * 100, 4),
        concentration_hhi=round(hhi, 6),
        largest_position_symbol=largest_symbol,
        largest_position_weight=round(float(weights[largest_symbol]), 6),
        positions=position_results,
        allocations=allocations,
        value_series=[
            SeriesPoint(date=index.date(), value_eur=money(float(value)))
            for index, value in sampled_series.items()
        ],
        warnings=[
            "Historical risk uses current holdings and is not actual account performance.",
            "Security prices are deterministic synthetic demo data.",
        ],
    )
