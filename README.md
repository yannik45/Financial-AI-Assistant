# Financial AI Assistant

Local-first portfolio intelligence platform with deterministic analytics, a
FastAPI API, and a React dashboard. Phase 1 intentionally uses synthetic market
prices and bundled ECB-style FX fixtures so the demo is reproducible and never
mistaken for live financial data.

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js 24 LTS and npm

## Run locally

```powershell
uv sync --all-groups
uv run alembic upgrade head
uv run financial-ai-api
```

In a second terminal:

```powershell
cd apps/web
npm install
npm run dev
```

Open `http://localhost:5173`. API documentation is available at
`http://localhost:8000/docs`.

## Transaction API

The local API seeds deterministic demo data for checking, savings, and
brokerage accounts. Transaction amounts are signed cash flows: money received
is positive, while spending, withdrawals, fees, and security purchases are
negative.

Available endpoints:

- `GET /v1/accounts`
- `GET /v1/accounts/{account_id}`
- `GET /v1/transactions`
- `GET /v1/transactions/{transaction_id}`
- `POST /v1/transactions`

The transaction list supports account, type, category, date-range, limit, and
offset filters. Security buy and sell requests require a brokerage account,
symbol, quantity, and unit price. All bundled transactions and counterparties
are synthetic demo data.

## Tests

```powershell
uv run pytest
cd apps/web
npm test
npm run build
```

The application is educational software, not financial advice. All included
security prices are synthetic demo data.
