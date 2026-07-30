# Financial AI Assistant

Local-first portfolio intelligence platform with deterministic analytics, a
FastAPI API, and a React dashboard. Phase 1 intentionally uses synthetic market
prices and bundled ECB-style FX fixtures so the demo is reproducible and never
mistaken for live financial data.

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js 22 or newer and npm

## Run locally

```powershell
python -m uv sync --all-groups
python -m uv run alembic upgrade head
python -m uv run financial-ai-api
```

In a second terminal:

```powershell
cd apps/web
npm.cmd install
npm.cmd run dev
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

## Documentation

Detailed technical and ML documentation is maintained under `docs/`:

- [Transaction category taxonomy](docs/ml/transaction_categories.md) defines
  the versioned expense labels, scope, examples, and boundary rules.
- [Transaction classification methodology](docs/ml/transaction_classification.md)
  records dataset provenance, preprocessing decisions, assumptions, evaluation
  protocol, baseline results, limitations, and planned model development.

New documentation should be added to this index when it is introduced so the
repository documentation remains discoverable from the project entry point.

## Tests

```powershell
python -m uv run pytest
cd apps/web
npm.cmd test
npm.cmd run build
```

`python -m uv` also works when the `uv` executable is not yet available on the
current terminal's `PATH`. Use `npm.cmd` in PowerShell environments where the
execution policy blocks `npm.ps1`. Generated local content in `.venv`,
`node_modules`, `dist`, and `data/runtime` is excluded from version control.

The application is educational software, not financial advice. All included
security prices are synthetic demo data.
