# Financial AI Assistant

A production-oriented portfolio project for deterministic financial analytics
and applied ML. It combines a FastAPI backend, React dashboard, SQLite ledger,
bilingual transaction categorization, guarded feedback lifecycle, and a
reproducible Docker Compose stack.

The project demonstrates production-oriented architecture and engineering
practices, but is not currently intended for production use or real customer
data. Security prices, demo portfolios, transactions, and ML training data are
synthetic. The stored ECB FX snapshot is the only real market reference data.
Nothing in this repository is financial advice.

## What is implemented

| Area | Current capability |
|---|---|
| Portfolio analytics | Valuation, allocation, P&L, return, volatility, drawdown, concentration, time series, and CSV portfolio import |
| Market data | Provider-neutral instrument search and cached daily history with source and freshness metadata; deterministic demo mode and optional Twelve Data adapter |
| Paper trading | Local simulated portfolios with server-priced buy/sell orders, cash controls, derived holdings, and realized/unrealized P&L |
| Transaction ledger | Checking, savings, and brokerage accounts; filters and manual transaction entry |
| Classification | Editable English/German category suggestions using auditable rules plus character TF-IDF and Logistic Regression |
| ML lifecycle | Frozen evaluation sets, abstention metrics, feedback capture, immutable exports, candidate gates, explicit promotion, and rollback artifacts |
| Delivery | Backend/frontend tests, GitHub Actions, multi-stage images, health checks, reverse proxy, and persistent Compose storage |

Financial metrics are calculated only by deterministic backend code. A future
LLM may select tested tools and explain their output, but must never calculate
financial metrics itself. RAG and LLM integration are not implemented yet.

## Architecture

```text
Browser -> React / Nginx -> FastAPI -> SQLite + versioned local ML artifacts
                              |----> deterministic portfolio analytics
                              `----> rules + bilingual expense classifier
```

See the [system overview](docs/architecture/system-overview.md) for component
boundaries and data flows. Container details are in the
[container architecture](docs/architecture/containerization.md).

## Quick start with Docker

Requires Docker with Compose:

```powershell
docker compose up --build --wait
```

- Dashboard: `http://localhost:5173`
- API health: `http://localhost:8000/health`
- OpenAPI documentation: `http://localhost:8000/docs`

Stop the stack without deleting its database or model artifacts:

```powershell
docker compose down
```

`docker compose down --volumes` also permanently removes the
container-managed runtime data.

## Local development

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 22+, and npm.
From the repository root:

```powershell
uv sync --all-groups
uv run alembic upgrade head
uv run financial-ai-bootstrap-category-model
uv run financial-ai-api
```

In a second terminal:

```powershell
cd apps/web
npm.cmd install
npm.cmd run dev
```

If `uv` is not available but the synchronized `.venv` already exists:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m financial_ai.ml.category_bootstrap
.\.venv\Scripts\python.exe -m uvicorn financial_ai.main:app --host 127.0.0.1 --port 8000 --reload
```

This fallback cannot install or update dependencies. Copy `.env.example` to
`.env` for backend overrides and `apps/web/.env.example` to `apps/web/.env` for
frontend overrides. Do not commit secrets.

The default `demo` market-data provider requires no credentials and keeps tests
reproducible. To explore real instruments through Twelve Data, set
`FINANCIAL_AI_MARKET_DATA_PROVIDER=twelve_data` and provide
`FINANCIAL_AI_MARKET_DATA_API_KEY`. External data remains subject to the
provider's plan, freshness, and usage terms; API keys stay server-side.

Paper portfolios use a separate immutable simulated-trade ledger. Orders are
priced by the backend from the latest cached daily close, never by a
browser-supplied price. Cash, holdings, average cost, and P&L are derived from
that ledger. No real brokerage order is placed.

## Verification

```powershell
uv run pytest
cd apps/web
npm.cmd test
npm.cmd run build
```

CI runs backend lint/tests, frontend tests/build, image builds, health checks,
and an end-to-end proxy smoke test. Generated content in `.venv`,
`node_modules`, `dist`, and `data/runtime` is ignored.

## Documentation

- [System overview](docs/architecture/system-overview.md): architecture,
  responsibilities, data flows, and current boundaries.
- [Container architecture](docs/architecture/containerization.md): images,
  networking, persistence, and CI checks.
- [ML documentation index](docs/ml/README.md): active contracts and frozen
  experiment records.
- [Data notes](data/README.md): synthetic data boundaries and ECB provenance.

The next major product increment is transaction fraud/risk scoring. RAG with
deterministic tool calling follows after the local product and ML foundations
are stable; cloud deployment remains a later phase.
