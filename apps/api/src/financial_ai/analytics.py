from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from financial_ai.market_data import (
    ASSETS,
    DATA_VERSION,
    DemoMarketDataProvider,
    market_data_provider,
)
from financial_ai.models import Portfolio
from financial_ai.schemas import (
    AllocationItem,
    AnalyticsResponse,
    PositionAnalytics,
    SeriesPoint,
)

MONEY = Decimal("0.01")

if TYPE_CHECKING:
    from financial_ai.market_data_service import MarketDataService


class AnalyticsError(ValueError):
    pass


@dataclass
class CurrentHolding:
    symbol: str
    quantity: Decimal
    book_cost: Decimal
    purchase_date: date
    asset_class: str
    sector: str
    region: str
    currency: str
    instrument_id: str | None = None


def money(value: float) -> Decimal:
    return Decimal(str(value)).quantize(MONEY, rounding=ROUND_HALF_UP)


def calculate_analytics(
    portfolio: Portfolio,
    provider: DemoMarketDataProvider = market_data_provider,
    market_service: "MarketDataService | None" = None,
) -> AnalyticsResponse:
    holdings = current_holdings(portfolio)
    if not holdings:
        raise AnalyticsError("Portfolio has no positions")

    symbols = sorted({holding.symbol for holding in holdings})
    prices = _price_history(holdings, provider, market_service).tail(252)
    if len(prices.dropna()) < 60:
        raise AnalyticsError("At least 60 common price observations are required")

    eur_prices = prices.copy()
    for symbol in symbols:
        currency = next(item.currency for item in holdings if item.symbol == symbol)
        fx = pd.Series(
            [provider.fx_on_or_before(currency, item.date()) for item in prices.index],
            index=prices.index,
        )
        eur_prices[symbol] = prices[symbol] * fx

    quantities = pd.Series(
        {
            symbol: sum(float(item.quantity) for item in holdings if item.symbol == symbol)
            for symbol in symbols
        }
    )
    value_series = eur_prices.mul(quantities, axis=1)
    portfolio_series = value_series.sum(axis=1)
    current_by_symbol = value_series.iloc[-1]
    market_value = float(current_by_symbol.sum())

    cost_by_symbol: dict[str, float] = defaultdict(float)
    for holding in holdings:
        purchase_fx = provider.fx_on_or_before(holding.currency, holding.purchase_date)
        cost_by_symbol[holding.symbol] += float(holding.book_cost) * purchase_fx
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
        for holding in holdings:
            symbol_total_quantity = quantities[holding.symbol]
            position_share = float(holding.quantity) / symbol_total_quantity
            grouped[getter(holding)] += float(current_by_symbol[holding.symbol]) * position_share
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
            "Historical risk reconstructs current ledger-derived holdings backwards and is not "
            "actual account performance.",
            "Demo instruments use deterministic synthetic prices; external observations expose "
            "their configured provider source.",
        ],
    )


def current_holdings(portfolio: Portfolio) -> list[CurrentHolding]:
    """Replay opening positions and portfolio-linked market orders using average cost."""
    states: dict[str, CurrentHolding] = {}
    for position in portfolio.positions:
        state = states.get(position.symbol)
        if state is None:
            states[position.symbol] = CurrentHolding(
                symbol=position.symbol,
                quantity=position.quantity,
                book_cost=position.quantity * position.purchase_price,
                purchase_date=position.purchase_date,
                asset_class=position.asset_class,
                sector=position.sector,
                region=position.region,
                currency=position.currency,
            )
        else:
            state.quantity += position.quantity
            state.book_cost += position.quantity * position.purchase_price
            state.purchase_date = min(state.purchase_date, position.purchase_date)

    if portfolio.account is None:
        return list(states.values())
    trades = sorted(
        (
            item
            for item in portfolio.account.transactions
            if item.market_instrument_id
            and item.transaction_type in {"security_buy", "security_sell"}
        ),
        key=lambda item: (item.booked_at, item.created_at),
    )
    for trade in trades:
        instrument = trade.market_instrument
        if instrument is None or trade.quantity is None or trade.unit_price is None:
            continue
        state = states.get(instrument.symbol)
        if trade.transaction_type == "security_buy":
            if state is None:
                demo_asset = ASSETS.get(instrument.symbol)
                state = CurrentHolding(
                    symbol=instrument.symbol,
                    quantity=Decimal("0"),
                    book_cost=Decimal("0"),
                    purchase_date=trade.booked_at,
                    asset_class=instrument.asset_class,
                    sector=demo_asset.sector if demo_asset else instrument.asset_class,
                    region=instrument.region or "Unknown",
                    currency=instrument.currency,
                    instrument_id=instrument.id,
                )
                states[instrument.symbol] = state
            state.quantity += trade.quantity
            state.book_cost += trade.quantity * trade.unit_price + trade.fees
            state.instrument_id = instrument.id
            state.purchase_date = min(state.purchase_date, trade.booked_at)
            continue
        if state is None or trade.quantity > state.quantity:
            raise AnalyticsError("Stored transactions would create a short position")
        average_cost = state.book_cost / state.quantity
        state.quantity -= trade.quantity
        state.book_cost -= average_cost * trade.quantity
        state.instrument_id = instrument.id
        if state.quantity == 0:
            del states[instrument.symbol]
    return list(states.values())


def _price_history(
    holdings: list[CurrentHolding],
    provider: DemoMarketDataProvider,
    market_service: "MarketDataService | None",
) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    demo_symbols = [item.symbol for item in holdings if item.symbol in ASSETS]
    if demo_symbols:
        demo_prices = provider.prices(demo_symbols)
        series.update({symbol: demo_prices[symbol] for symbol in demo_symbols})
    for holding in holdings:
        if holding.symbol in series:
            continue
        if market_service is None or holding.instrument_id is None:
            raise AnalyticsError(f"No historical price service is available for {holding.symbol}")
        history = market_service.history(holding.instrument_id)
        series[holding.symbol] = pd.Series(
            [float(item.adjusted_close or item.close) for item in history.points],
            index=pd.to_datetime([item.observed_on for item in history.points]),
        )
    return pd.concat(series, axis=1, join="inner").sort_index()
