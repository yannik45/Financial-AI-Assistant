import json
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from financial_ai.analytics import AnalyticsError, calculate_analytics
from financial_ai.config import get_settings
from financial_ai.database import Base, SessionLocal, engine, get_session
from financial_ai.importer import PortfolioImportError, parse_portfolio_csv
from financial_ai.market_data import market_data_provider
from financial_ai.models import Portfolio
from financial_ai.schemas import AnalyticsResponse, CatalogAsset, PortfolioRead, PortfolioSummary
from financial_ai.seed import seed_demo_portfolios

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("financial_ai")


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        seed_demo_portfolios(session)
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
    logger.info(json.dumps({"event": "request_completed", "correlation_id": correlation_id, "method": request.method, "path": request.url.path, "status": response.status_code, "duration_ms": elapsed_ms}))
    return response


@app.exception_handler(PortfolioImportError)
async def import_error_handler(_: Request, exc: PortfolioImportError):
    return JSONResponse(status_code=422, content={"code": "invalid_portfolio_csv", "message": str(exc), "details": exc.details})


@app.get("/health")
def health(session: Session = Depends(get_session)) -> dict[str, str]:
    session.execute(text("SELECT 1"))
    return {"status": "ok", "database": "ok"}


@app.get("/v1/market/catalog", response_model=list[CatalogAsset])
def market_catalog() -> list[CatalogAsset]:
    return [CatalogAsset(**asset.__dict__) for asset in market_data_provider.catalog()]


@app.get("/v1/portfolios", response_model=list[PortfolioSummary])
def list_portfolios(session: Session = Depends(get_session)) -> list[PortfolioSummary]:
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
        raise HTTPException(status_code=404, detail={"code": "portfolio_not_found", "message": "Portfolio not found"})
    return portfolio


@app.get("/v1/portfolios/{portfolio_id}", response_model=PortfolioRead)
def get_portfolio(portfolio_id: str, session: Session = Depends(get_session)) -> Portfolio:
    return get_portfolio_or_404(portfolio_id, session)


@app.post("/v1/portfolios/import", response_model=PortfolioRead, status_code=201)
async def import_portfolio(name: str = Form(min_length=1, max_length=120), file: UploadFile = File(), session: Session = Depends(get_session)) -> Portfolio:
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
def portfolio_analytics(portfolio_id: str, session: Session = Depends(get_session)) -> AnalyticsResponse:
    portfolio = get_portfolio_or_404(portfolio_id, session)
    try:
        return calculate_analytics(portfolio)
    except AnalyticsError as exc:
        raise HTTPException(status_code=422, detail={"code": "analytics_unavailable", "message": str(exc)}) from exc


def run() -> None:
    import uvicorn
    uvicorn.run("financial_ai.main:app", host="127.0.0.1", port=8000, reload=True)
