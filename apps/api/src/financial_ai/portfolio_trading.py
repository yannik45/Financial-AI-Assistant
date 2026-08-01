from dataclasses import dataclass
from datetime import UTC
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from financial_ai.market_data_service import MarketDataService
from financial_ai.models import Account, MarketInstrument, Portfolio, Transaction
from financial_ai.schemas import (
    MarketQuoteRead,
    PortfolioCreate,
    PortfolioHoldingRead,
    PortfolioOrderCreate,
    PortfolioTradeRead,
    TradeSide,
    TradingPortfolioRead,
    TradingPortfolioSummary,
)

MONEY = Decimal("0.01")
PRICE = Decimal("0.00000001")
ZERO = Decimal("0")


class PortfolioTradingError(ValueError):
    code = "portfolio_trading_error"


class PortfolioNotFoundError(PortfolioTradingError):
    code = "portfolio_not_found"


class PortfolioAccountMissingError(PortfolioTradingError):
    code = "portfolio_account_missing"


class InsufficientCashError(PortfolioTradingError):
    code = "insufficient_cash"


class InsufficientHoldingsError(PortfolioTradingError):
    code = "insufficient_holdings"


class CurrencyMismatchError(PortfolioTradingError):
    code = "portfolio_currency_mismatch"


class IdempotencyConflictError(PortfolioTradingError):
    code = "order_idempotency_conflict"


@dataclass
class HoldingState:
    quantity: Decimal = ZERO
    book_cost: Decimal = ZERO
    realized_pnl: Decimal = ZERO


def money(value: Decimal) -> Decimal:
    return value.quantize(MONEY, rounding=ROUND_HALF_UP)


def price(value: Decimal) -> Decimal:
    return value.quantize(PRICE, rounding=ROUND_HALF_UP)


class PortfolioTradingService:
    """Treat a portfolio and its brokerage account as one aggregate."""

    def __init__(self, session: Session, market_data: MarketDataService) -> None:
        self._session = session
        self._market_data = market_data

    def create_portfolio(self, payload: PortfolioCreate) -> TradingPortfolioRead:
        account = Account(
            name=f"{payload.name} Brokerage",
            account_type="brokerage",
            currency=payload.base_currency,
            kind="manual",
            opening_balance=payload.starting_cash,
        )
        portfolio = Portfolio(
            name=payload.name,
            base_currency=payload.base_currency,
            kind="manual",
            account=account,
        )
        self._session.add(portfolio)
        self._session.commit()
        return self.detail(portfolio.id)

    def list_portfolios(self) -> list[TradingPortfolioSummary]:
        portfolios = self._session.scalars(
            select(Portfolio).order_by(Portfolio.kind, Portfolio.name)
        ).all()
        return [self._summary(item) for item in portfolios]

    def execute_order(self, portfolio_id: str, payload: PortfolioOrderCreate) -> PortfolioTradeRead:
        portfolio = self._get_portfolio(portfolio_id)
        account = self._get_account(portfolio)
        existing = self._session.scalar(
            select(Transaction).where(
                Transaction.account_id == account.id,
                Transaction.client_order_id == payload.client_order_id,
            )
        )
        if existing:
            expected_type = f"security_{payload.side.value}"
            if (
                existing.market_instrument_id != payload.instrument_id
                or existing.transaction_type != expected_type
                or existing.quantity != payload.quantity
            ):
                raise IdempotencyConflictError(
                    "The client_order_id is already associated with a different order"
                )
            return self._trade_read(existing)

        quote = self._market_data.quote(payload.instrument_id)
        if quote.instrument.currency != portfolio.base_currency:
            raise CurrencyMismatchError("The instrument currency must match the portfolio currency")
        states, cash = self._replay(portfolio)
        state = states.get(payload.instrument_id, HoldingState())
        notional = money(payload.quantity * quote.close)
        if payload.side == TradeSide.BUY and notional > cash:
            raise InsufficientCashError(
                f"Order requires {notional} {portfolio.base_currency}, "
                f"but only {money(cash)} is available"
            )
        if payload.side == TradeSide.SELL and payload.quantity > state.quantity:
            raise InsufficientHoldingsError(
                f"Cannot sell {payload.quantity}; current holding is {state.quantity}"
            )

        instrument = self._session.get(MarketInstrument, payload.instrument_id)
        if instrument is None:
            raise PortfolioTradingError("The selected market instrument no longer exists")
        signed_amount = -notional if payload.side == TradeSide.BUY else notional
        transaction = Transaction(
            account_id=account.id,
            booked_at=quote.observed_on,
            name=f"{payload.side.value.title()} {instrument.symbol}",
            amount=signed_amount,
            currency=portfolio.base_currency,
            transaction_type=f"security_{payload.side.value}",
            category="Investments",
            source="market_order",
            market_instrument_id=instrument.id,
            client_order_id=payload.client_order_id,
            security_symbol=instrument.symbol,
            quantity=payload.quantity,
            unit_price=quote.close,
            fees=ZERO,
            taxes=ZERO,
            price_observed_on=quote.observed_on,
            price_source=quote.source,
        )
        self._session.add(transaction)
        self._session.commit()
        self._session.refresh(transaction)
        return self._trade_read(transaction)

    def detail(self, portfolio_id: str) -> TradingPortfolioRead:
        portfolio = self._get_portfolio(portfolio_id)
        account = self._get_account(portfolio)
        states, cash = self._replay(portfolio)
        valued: list[tuple[HoldingState, MarketQuoteRead]] = []
        warnings: list[str] = []
        for instrument_id, state in states.items():
            if state.quantity > ZERO:
                valued.append((state, self._market_data.quote(instrument_id)))
        holdings_value = sum((state.quantity * quote.close for state, quote in valued), start=ZERO)
        holdings = [
            PortfolioHoldingRead(
                instrument=quote.instrument,
                quantity=state.quantity,
                average_cost=price(state.book_cost / state.quantity),
                latest_price=quote.close,
                market_value=money(state.quantity * quote.close),
                unrealized_pnl=money(state.quantity * quote.close - state.book_cost),
                weight=round(float(state.quantity * quote.close / holdings_value), 6)
                if holdings_value
                else 0,
                price_observed_on=quote.observed_on,
                price_source=quote.source,
                quote_is_stale=quote.is_stale,
            )
            for state, quote in sorted(
                valued, key=lambda item: item[0].quantity * item[1].close, reverse=True
            )
        ]
        trades = [
            item
            for item in account.transactions
            if item.transaction_type in {"security_buy", "security_sell"}
            and item.market_instrument_id
        ]
        opening_investment = sum(
            (position.quantity * position.purchase_price for position in portfolio.positions),
            start=ZERO,
        )
        total_equity = cash + holdings_value
        realized_pnl = sum((state.realized_pnl for state in states.values()), start=ZERO)
        if portfolio.positions:
            warnings.append(
                "Imported and demo positions are opening holdings; subsequent orders and "
                "cash movements are ledger transactions."
            )
        if any(item.quote_is_stale for item in holdings):
            warnings.append("At least one holding uses a stale cached quote.")
        warnings.extend(
            [
                "Simulation only: no real order is placed.",
                "Orders use the latest cached daily close; fees, taxes, spreads, and "
                "slippage are zero.",
            ]
        )
        return TradingPortfolioRead(
            **self._summary(portfolio).model_dump(),
            cash_balance=money(cash),
            holdings_value=money(holdings_value),
            total_equity=money(total_equity),
            total_pnl=money(total_equity - account.opening_balance - opening_investment),
            realized_pnl=money(realized_pnl),
            holdings=holdings,
            trades=[self._trade_read(item) for item in trades],
            warnings=warnings,
        )

    def _summary(self, portfolio: Portfolio) -> TradingPortfolioSummary:
        account = self._get_account(portfolio)
        trade_count = sum(
            item.transaction_type in {"security_buy", "security_sell"}
            for item in account.transactions
        )
        return TradingPortfolioSummary(
            id=portfolio.id,
            name=portfolio.name,
            base_currency=portfolio.base_currency,
            opening_cash=account.opening_balance,
            created_at=portfolio.created_at,
            trade_count=trade_count,
        )

    def _get_portfolio(self, portfolio_id: str) -> Portfolio:
        portfolio = self._session.get(Portfolio, portfolio_id)
        if portfolio is None:
            raise PortfolioNotFoundError("Portfolio not found")
        return portfolio

    @staticmethod
    def _get_account(portfolio: Portfolio) -> Account:
        if portfolio.account is None:
            raise PortfolioAccountMissingError("Portfolio has no linked brokerage account")
        return portfolio.account

    def _replay(self, portfolio: Portfolio) -> tuple[dict[str, HoldingState], Decimal]:
        account = self._get_account(portfolio)
        cash = account.opening_balance + sum(
            (item.amount for item in account.transactions), start=ZERO
        )
        states: dict[str, HoldingState] = {}
        for position in portfolio.positions:
            instrument = self._instrument_for_symbol(position.symbol)
            if instrument:
                state = states.setdefault(instrument.id, HoldingState())
                state.quantity += position.quantity
                state.book_cost += position.quantity * position.purchase_price
        for transaction in sorted(
            account.transactions, key=lambda item: (item.booked_at, item.created_at)
        ):
            if not transaction.market_instrument_id or transaction.transaction_type not in {
                "security_buy",
                "security_sell",
            }:
                continue
            state = states.setdefault(transaction.market_instrument_id, HoldingState())
            assert transaction.quantity is not None and transaction.unit_price is not None
            notional = transaction.quantity * transaction.unit_price
            if transaction.transaction_type == "security_buy":
                state.quantity += transaction.quantity
                state.book_cost += notional + transaction.fees
            else:
                if transaction.quantity > state.quantity:
                    raise PortfolioTradingError("Stored transactions would create a short position")
                average_cost = state.book_cost / state.quantity
                released_cost = average_cost * transaction.quantity
                state.quantity -= transaction.quantity
                state.book_cost -= released_cost
                state.realized_pnl += (
                    notional - transaction.fees - transaction.taxes - released_cost
                )
                if state.quantity == ZERO:
                    state.book_cost = ZERO
        return states, cash

    def _instrument_for_symbol(self, symbol: str) -> MarketInstrument | None:
        instrument = self._session.scalar(
            select(MarketInstrument).where(MarketInstrument.symbol == symbol).limit(1)
        )
        if instrument:
            return instrument
        matches = self._market_data.search(symbol, limit=10)
        return next((item for item in matches if item.symbol == symbol), None)

    @staticmethod
    def _trade_read(transaction: Transaction) -> PortfolioTradeRead:
        side = TradeSide.BUY if transaction.transaction_type == "security_buy" else TradeSide.SELL
        assert transaction.market_instrument is not None
        assert transaction.client_order_id is not None
        assert transaction.quantity is not None and transaction.unit_price is not None
        assert transaction.price_observed_on is not None and transaction.price_source is not None
        return PortfolioTradeRead(
            id=transaction.id,
            client_order_id=transaction.client_order_id,
            side=side,
            quantity=transaction.quantity,
            unit_price=transaction.unit_price,
            fees=transaction.fees,
            currency=transaction.currency,
            price_observed_on=transaction.price_observed_on,
            price_source=transaction.price_source,
            executed_at=transaction.created_at.replace(tzinfo=UTC),
            instrument=transaction.market_instrument,
        )
