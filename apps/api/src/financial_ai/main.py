import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from financial_ai.analytics import AnalyticsError, calculate_analytics
from financial_ai.config import get_settings
from financial_ai.database import SessionLocal, get_session
from financial_ai.importer import PortfolioImportError, parse_portfolio_csv
from financial_ai.market_data import market_data_provider
from financial_ai.ml.category_artifact import ModelArtifactError
from financial_ai.ml.category_service import (
    TransactionClassification,
    TransactionClassifier,
    get_transaction_classifier,
)
from financial_ai.ml.transaction_classification import (
    TAXONOMY_VERSION,
    ClassificationMethod,
    determine_feedback_status,
    route_transaction_text,
)
from financial_ai.models import Account, Portfolio, Transaction, TransactionClassificationRecord
from financial_ai.schemas import (
    AccountRead,
    AnalyticsResponse,
    CatalogAsset,
    PortfolioRead,
    PortfolioSummary,
    TransactionClassificationRequest,
    TransactionClassificationResponse,
    TransactionCreate,
    TransactionPage,
    TransactionRead,
    TransactionType,
)
from financial_ai.seed import seed_demo_accounts, seed_demo_portfolios

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("financial_ai")
SessionDependency = Annotated[Session, Depends(get_session)]
ClassifierDependency = Annotated[TransactionClassifier, Depends(get_transaction_classifier)]


@asynccontextmanager
async def lifespan(_: FastAPI):
    with SessionLocal() as session:
        seed_demo_portfolios(session)
        seed_demo_accounts(session)
    yield


app = FastAPI(title="Financial AI Assistant API", version="0.1.0", lifespan=lifespan)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(json.dumps({"event": "request_failed", "correlation_id": correlation_id}))
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    logger.info(
        json.dumps(
            {
                "event": "request_completed",
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": elapsed_ms,
            }
        )
    )
    return response


@app.exception_handler(PortfolioImportError)
async def import_error_handler(_: Request, exc: PortfolioImportError):
    return JSONResponse(
        status_code=422,
        content={"code": "invalid_portfolio_csv", "message": str(exc), "details": exc.details},
    )


@app.get("/health")
def health(session: SessionDependency) -> dict[str, str]:
    session.execute(text("SELECT 1"))
    return {"status": "ok", "database": "ok"}


@app.get("/v1/market/catalog", response_model=list[CatalogAsset])
def market_catalog() -> list[CatalogAsset]:
    return [CatalogAsset(**asset.__dict__) for asset in market_data_provider.catalog()]


@app.get("/v1/portfolios", response_model=list[PortfolioSummary])
def list_portfolios(session: SessionDependency) -> list[PortfolioSummary]:
    portfolios = session.scalars(select(Portfolio).order_by(Portfolio.kind, Portfolio.name)).all()
    return [
        PortfolioSummary(
            id=portfolio.id,
            name=portfolio.name,
            base_currency=portfolio.base_currency,
            kind=portfolio.kind,
            created_at=portfolio.created_at,
            position_count=len(portfolio.positions),
        )
        for portfolio in portfolios
    ]


def get_portfolio_or_404(portfolio_id: str, session: Session) -> Portfolio:
    portfolio = session.get(Portfolio, portfolio_id)
    if portfolio is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "portfolio_not_found", "message": "Portfolio not found"},
        )
    return portfolio


@app.get("/v1/portfolios/{portfolio_id}", response_model=PortfolioRead)
def get_portfolio(portfolio_id: str, session: SessionDependency) -> Portfolio:
    return get_portfolio_or_404(portfolio_id, session)


@app.post("/v1/portfolios/import", response_model=PortfolioRead, status_code=201)
async def import_portfolio(
    name: Annotated[str, Form(min_length=1, max_length=120)],
    file: Annotated[UploadFile, File()],
    session: SessionDependency,
) -> Portfolio:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise PortfolioImportError([{"field": "file", "message": "A .csv file is required"}])
    content = await file.read()
    if len(content) > 1_000_000:
        raise PortfolioImportError([{"field": "file", "message": "CSV must not exceed 1 MB"}])
    portfolio = parse_portfolio_csv(content, name)
    session.add(portfolio)
    session.commit()
    session.refresh(portfolio)
    return portfolio


@app.get("/v1/portfolios/{portfolio_id}/analytics", response_model=AnalyticsResponse)
def portfolio_analytics(portfolio_id: str, session: SessionDependency) -> AnalyticsResponse:
    portfolio = get_portfolio_or_404(portfolio_id, session)
    try:
        return calculate_analytics(portfolio)
    except AnalyticsError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "analytics_unavailable", "message": str(exc)}
        ) from exc


def get_account_or_404(account_id: str, session: Session) -> Account:
    account = session.get(Account, account_id)
    if account is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "account_not_found", "message": "Account not found"},
        )
    return account


@app.get("/v1/accounts", response_model=list[AccountRead])
def list_accounts(session: SessionDependency) -> list[AccountRead]:
    accounts = session.scalars(select(Account).order_by(Account.account_type, Account.name)).all()
    return [
        AccountRead(
            id=account.id,
            name=account.name,
            account_type=account.account_type,
            currency=account.currency,
            kind=account.kind,
            created_at=account.created_at,
            transaction_count=len(account.transactions),
        )
        for account in accounts
    ]


@app.get("/v1/accounts/{account_id}", response_model=AccountRead)
def get_account(account_id: str, session: SessionDependency) -> AccountRead:
    account = get_account_or_404(account_id, session)
    return AccountRead(
        id=account.id,
        name=account.name,
        account_type=account.account_type,
        currency=account.currency,
        kind=account.kind,
        created_at=account.created_at,
        transaction_count=len(account.transactions),
    )


@app.get("/v1/transactions", response_model=TransactionPage)
def list_transactions(
    session: SessionDependency,
    account_id: str | None = None,
    transaction_type: TransactionType | None = None,
    cash_flow: Literal["inflow", "outflow"] | None = None,
    category: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> TransactionPage:
    filters = []
    if account_id:
        get_account_or_404(account_id, session)
        filters.append(Transaction.account_id == account_id)
    if transaction_type:
        filters.append(Transaction.transaction_type == transaction_type.value)
    if cash_flow == "inflow":
        filters.append(Transaction.amount > 0)
    elif cash_flow == "outflow":
        filters.append(Transaction.amount < 0)
    if category:
        filters.append(func.lower(Transaction.category) == category.lower())
    if date_from:
        filters.append(Transaction.booked_at >= date_from)
    if date_to:
        filters.append(Transaction.booked_at <= date_to)
    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_date_range", "message": "date_from must not exceed date_to"},
        )

    total = session.scalar(select(func.count()).select_from(Transaction).where(*filters)) or 0
    statement = (
        select(Transaction)
        .where(*filters)
        .order_by(Transaction.booked_at.desc(), Transaction.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    items = list(session.scalars(statement).all())
    return TransactionPage(items=items, total=total, limit=limit, offset=offset)


@app.get("/v1/transactions/{transaction_id}", response_model=TransactionRead)
def get_transaction(transaction_id: str, session: SessionDependency) -> Transaction:
    transaction = session.get(Transaction, transaction_id)
    if transaction is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "transaction_not_found", "message": "Transaction not found"},
        )
    return transaction


@app.post(
    "/v1/transactions/classify",
    response_model=TransactionClassificationResponse,
)
def classify_transaction(
    payload: TransactionClassificationRequest,
    classifier: ClassifierDependency,
) -> TransactionClassificationResponse:
    try:
        result = classifier.classify(
            description=payload.description,
            amount=payload.amount,
            counterparty=payload.counterparty,
        )
    except ModelArtifactError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "category_model_unavailable", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "classification_input_invalid", "message": str(exc)},
        ) from exc
    return TransactionClassificationResponse(
        category=result.category,
        route=result.route,
        classification_method=result.method,
        confidence=result.confidence,
        needs_review=result.needs_review,
        reason=result.reason,
        taxonomy_version=result.taxonomy_version,
        model_version=result.model_version,
    )


@app.post("/v1/transactions", response_model=TransactionRead, status_code=201)
def create_transaction(
    payload: TransactionCreate,
    session: SessionDependency,
    classifier: ClassifierDependency,
) -> Transaction:
    account = get_account_or_404(payload.account_id, session)
    security_types = {TransactionType.SECURITY_BUY, TransactionType.SECURITY_SELL}
    if payload.transaction_type in security_types and account.account_type != "brokerage":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_account_type",
                "message": "Security transactions require a brokerage account",
            },
        )

    try:
        classification = classifier.classify(
            description=payload.name,
            amount=payload.amount,
            counterparty=payload.counterparty,
        )
    except ModelArtifactError:
        route = route_transaction_text(payload.name, payload.amount, payload.counterparty)
        classification = TransactionClassification(
            category=None,
            route=route.route,
            method=ClassificationMethod.NONE,
            confidence=None,
            needs_review=True,
            reason="Category model artifact was unavailable when the transaction was saved.",
            taxonomy_version=TAXONOMY_VERSION,
            model_version=None,
        )

    transaction = Transaction(
        **payload.model_dump(
            mode="python",
            exclude={
                "currency",
                "transaction_type",
                "security_symbol",
                "category_confirmed",
            },
        ),
        currency=payload.currency.upper(),
        transaction_type=payload.transaction_type.value,
        security_symbol=payload.security_symbol.upper() if payload.security_symbol else None,
        source="manual",
    )
    session.add(transaction)
    session.flush()
    transaction.classifications.append(
        TransactionClassificationRecord(
            transaction_id=transaction.id,
            predicted_category=classification.category,
            final_category=payload.category,
            route=classification.route.value,
            classification_method=classification.method.value,
            confidence=classification.confidence,
            needs_review=classification.needs_review,
            feedback_status=determine_feedback_status(
                classification.category,
                payload.category,
                payload.category_confirmed,
            ).value,
            reason=classification.reason,
            taxonomy_version=classification.taxonomy_version,
            model_version=classification.model_version,
        )
    )
    session.commit()
    session.refresh(transaction)
    return transaction


def run() -> None:
    import uvicorn

    uvicorn.run("financial_ai.main:app", host="127.0.0.1", port=8000, reload=True)
