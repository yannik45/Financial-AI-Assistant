import csv
import io
from datetime import date
from decimal import Decimal, InvalidOperation

from financial_ai.market_data import ASSETS
from financial_ai.models import Portfolio, Position

EXPECTED_COLUMNS = [
    "symbol",
    "quantity",
    "purchase_price",
    "purchase_date",
    "asset_class",
    "sector",
    "region",
    "currency",
]


class PortfolioImportError(ValueError):
    def __init__(self, details: list[dict[str, object]]) -> None:
        super().__init__("Portfolio CSV validation failed")
        self.details = details


def parse_portfolio_csv(content: bytes, name: str) -> Portfolio:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PortfolioImportError(
            [{"field": "file", "message": "CSV must be UTF-8 encoded"}]
        ) from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != EXPECTED_COLUMNS:
        raise PortfolioImportError(
            [{"field": "header", "message": f"Expected columns: {', '.join(EXPECTED_COLUMNS)}"}]
        )

    positions: list[Position] = []
    errors: list[dict[str, object]] = []
    for row_number, row in enumerate(reader, start=2):
        symbol = row["symbol"].strip().upper()
        asset = ASSETS.get(symbol)
        if asset is None:
            errors.append(
                {"row": row_number, "field": "symbol", "message": f"Unknown symbol: {symbol}"}
            )
            continue
        try:
            quantity = Decimal(row["quantity"])
            price = Decimal(row["purchase_price"])
            purchase_date = date.fromisoformat(row["purchase_date"])
            if quantity <= 0 or price <= 0:
                raise ValueError("quantity and purchase_price must be positive")
        except (InvalidOperation, ValueError) as exc:
            errors.append({"row": row_number, "field": "numeric_or_date", "message": str(exc)})
            continue
        supplied = (row["asset_class"], row["sector"], row["region"], row["currency"].upper())
        expected = (asset.asset_class, asset.sector, asset.region, asset.currency)
        if supplied != expected:
            errors.append(
                {
                    "row": row_number,
                    "field": "metadata",
                    "message": "Asset metadata does not match the market catalog",
                }
            )
            continue
        positions.append(
            Position(
                symbol=symbol,
                quantity=quantity,
                purchase_price=price,
                purchase_date=purchase_date,
                asset_class=asset.asset_class,
                sector=asset.sector,
                region=asset.region,
                currency=asset.currency,
            )
        )
    if not positions and not errors:
        errors.append({"field": "file", "message": "CSV contains no positions"})
    if errors:
        raise PortfolioImportError(errors)
    return Portfolio(name=name.strip(), kind="imported", positions=positions)
