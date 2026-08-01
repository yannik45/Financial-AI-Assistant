from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from financial_ai.market_data_service import MarketDataService
from financial_ai.models import MarketInstrument, PaperPortfolio, PaperTrade
from financial_ai.schemas import (
    MarketQuoteRead,
    PaperHoldingRead,
    PaperOrderCreate,
    PaperPortfolioCreate,
    PaperPortfolioRead,
    PaperPortfolioSummary,
    PaperTradeRead,
    PaperTradeSide,
)

MONEY = Decimal("0.01")
PRICE = Decimal("0.00000001")
ZERO = Decimal("0")


class PaperTradingError(ValueError):
    code = "paper_trading_error"


class PaperPortfolioNotFoundError(PaperTradingError):
    code = "paper_portfolio_not_found"


class InsufficientCashError(PaperTradingError):
    code = "insufficient_paper_cash"


class InsufficientHoldingsError(PaperTradingError):
    code = "insufficient_paper_holdings"


class CurrencyMismatchError(PaperTradingError):
    code = "paper_currency_mismatch"


class IdempotencyConflictError(PaperTradingError):
    code = "paper_order_idempotency_conflict"


@dataclass
class HoldingState:
    quantity: Decimal = ZERO
    book_cost: Decimal = ZERO
    realized_pnl: Decimal = ZERO


def money(value: Decimal) -> Decimal:
    return value.quantize(MONEY, rounding=ROUND_HALF_UP)


def price(value: Decimal) -> Decimal:
    return value.quantize(PRICE, rounding=ROUND_HALF_UP)


class PaperTradingService:
    def __init__(self, session: Session, market_data: MarketDataService) -> None:
        self._session = session
        self._market_data = market_data

    def create_portfolio(self, payload: PaperPortfolioCreate) -> PaperPortfolioRead:
        portfolio = PaperPortfolio(
            name=payload.name,
            base_currency=payload.base_currency,
            starting_cash=payload.starting_cash,
        )
        self._session.add(portfolio)
        self._session.commit()
        self._session.refresh(portfolio)
        return self.detail(portfolio.id)

    def list_portfolios(self) -> list[PaperPortfolioSummary]:
        portfolios = self._session.scalars(
            select(PaperPortfolio).order_by(PaperPortfolio.created_at, PaperPortfolio.name)
        ).all()
        return [
            PaperPortfolioSummary(
                id=item.id,
                name=item.name,
                base_currency=item.base_currency,
                starting_cash=item.starting_cash,
                created_at=item.created_at,
                trade_count=len(item.trades),
            )
            for item in portfolios
        ]

    def execute_order(self, portfolio_id: str, payload: PaperOrderCreate) -> PaperTradeRead:
        portfolio = self._get_portfolio(portfolio_id)
        existing = self._session.scalar(
            select(PaperTrade).where(
                PaperTrade.portfolio_id == portfolio.id,
                PaperTrade.client_order_id == payload.client_order_id,
            )
        )
        if existing:
            if (
                existing.instrument_id != payload.instrument_id
                or existing.side != payload.side.value
                or existing.quantity != payload.quantity
            ):
                raise IdempotencyConflictError(
                    "The client_order_id is already associated with a different paper order"
                )
            return PaperTradeRead.model_validate(existing)

        quote = self._market_data.quote(payload.instrument_id)
        if quote.instrument.currency != portfolio.base_currency:
            raise CurrencyMismatchError(
                "Paper portfolios currently accept instruments in their base currency only"
            )
        states, cash = self._replay(portfolio)
        state = states.get(payload.instrument_id, HoldingState())
        notional = payload.quantity * quote.close
        if payload.side == PaperTradeSide.BUY and notional > cash:
            raise InsufficientCashError(
                f"Paper order requires {money(notional)} {portfolio.base_currency}, "
                f"but only {money(cash)} is available"
            )
        if payload.side == PaperTradeSide.SELL and payload.quantity > state.quantity:
            raise InsufficientHoldingsError(
                f"Cannot sell {payload.quantity}; paper holding is {state.quantity}"
            )

        instrument = self._session.get(MarketInstrument, payload.instrument_id)
        if instrument is None:
            raise PaperTradingError("The selected market instrument no longer exists")
        trade = PaperTrade(
            portfolio_id=portfolio.id,
            instrument_id=instrument.id,
            client_order_id=payload.client_order_id,
            side=payload.side.value,
            quantity=payload.quantity,
            unit_price=quote.close,
            fees=Decimal("0.00"),
            currency=portfolio.base_currency,
            price_observed_on=quote.observed_on,
            price_source=quote.source,
        )
        self._session.add(trade)
        self._session.commit()
        self._session.refresh(trade)
        return PaperTradeRead.model_validate(trade)

    def detail(self, portfolio_id: str) -> PaperPortfolioRead:
        portfolio = self._get_portfolio(portfolio_id)
        states, cash = self._replay(portfolio)
        active = {instrument_id: state for instrument_id, state in states.items() if state.quantity}
        valued: list[tuple[HoldingState, MarketQuoteRead]] = []
        for instrument_id, state in active.items():
            valued.append((state, self._market_data.quote(instrument_id)))
        holdings_value = sum(
            (state.quantity * quote.close for state, quote in valued),
            start=ZERO,
        )
        holdings = [
            PaperHoldingRead(
                instrument=quote.instrument,
                quantity=state.quantity,
                average_cost=price(state.book_cost / state.quantity),
                latest_price=quote.close,
                market_value=money(state.quantity * quote.close),
                unrealized_pnl=money(state.quantity * quote.close - state.book_cost),
                weight=(
                    round(float(state.quantity * quote.close / holdings_value), 6)
                    if holdings_value
                    else 0
                ),
                price_observed_on=quote.observed_on,
                price_source=quote.source,
                quote_is_stale=quote.is_stale,
            )
            for state, quote in sorted(
                valued,
                key=lambda item: item[0].quantity * item[1].close,
                reverse=True,
            )
        ]
        realized_pnl = sum((state.realized_pnl for state in states.values()), start=ZERO)
        total_equity = cash + holdings_value
        warnings = [
            "Paper trading only: no real order is placed.",
            "Orders use the latest cached daily close; intraday execution is not simulated.",
            "Transaction fees, taxes, spreads, and slippage are currently zero.",
        ]
        if any(item.quote_is_stale for item in holdings):
            warnings.append("At least one holding uses a stale cached quote.")
        return PaperPortfolioRead(
            id=portfolio.id,
            name=portfolio.name,
            base_currency=portfolio.base_currency,
            starting_cash=portfolio.starting_cash,
            created_at=portfolio.created_at,
            trade_count=len(portfolio.trades),
            cash_balance=money(cash),
            holdings_value=money(holdings_value),
            total_equity=money(total_equity),
            total_pnl=money(total_equity - portfolio.starting_cash),
            realized_pnl=money(realized_pnl),
            holdings=holdings,
            trades=[PaperTradeRead.model_validate(item) for item in portfolio.trades],
            warnings=warnings,
        )

    def _get_portfolio(self, portfolio_id: str) -> PaperPortfolio:
        portfolio = self._session.get(PaperPortfolio, portfolio_id)
        if portfolio is None:
            raise PaperPortfolioNotFoundError("Paper portfolio not found")
        return portfolio

    @staticmethod
    def _replay(portfolio: PaperPortfolio) -> tuple[dict[str, HoldingState], Decimal]:
        cash = portfolio.starting_cash
        states: dict[str, HoldingState] = {}
        for trade in portfolio.trades:
            state = states.setdefault(trade.instrument_id, HoldingState())
            notional = trade.quantity * trade.unit_price
            if trade.side == PaperTradeSide.BUY:
                state.quantity += trade.quantity
                state.book_cost += notional + trade.fees
                cash -= notional + trade.fees
                continue
            if trade.quantity > state.quantity:
                raise PaperTradingError("Stored paper trades would create a short position")
            average_cost = state.book_cost / state.quantity
            released_cost = average_cost * trade.quantity
            state.quantity -= trade.quantity
            state.book_cost -= released_cost
            state.realized_pnl += notional - trade.fees - released_cost
            cash += notional - trade.fees
            if state.quantity == ZERO:
                state.book_cost = ZERO
        return states, cash
