# Financial AI Assistant

A full-stack portfolio intelligence platform that connects portfolio trading,
account cash flows, deterministic risk analytics, external market data, and
applied machine learning in one auditable system.

## At a glance

| Aspect | Summary |
|---|---|
| Product | Portfolio dashboard with linked brokerage accounts, simulated trading, transaction intelligence, risk analytics, and reproducible ML workflows |
| Problem | Financial data, portfolio state, and ML predictions are often handled in disconnected prototypes; this project keeps them consistent behind tested backend contracts |
| Engineering focus | Deterministic financial calculations, traceable data provenance, leakage-aware evaluation, human review, versioned artifacts, and containerized delivery |
| Current ML | Bilingual transaction classification and 20-day market-volatility forecasting with transparent statistical and regularized linear baselines |

## Technology stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite, TanStack Query, Recharts |
| Backend | Python 3.12, FastAPI, Pydantic, SQLAlchemy 2, Alembic |
| Data and ML | SQLite, pandas, NumPy, scikit-learn, versioned local artifacts |
| Delivery | uv, npm, pytest, Ruff, GitHub Actions, Docker Compose, unprivileged Nginx |

## What is implemented

| Area | Current capability |
|---|---|
| Portfolio analytics | Ledger-derived holdings, valuation, allocation, P&L, return, volatility, drawdown, separate market-risk, diversification, and liquidity indicators, time series, and CSV import |
| Market data | Provider-neutral instrument search and cached daily history with source and freshness metadata; deterministic demo mode and optional Alpaca adapter |
| Portfolio trading | Buy/sell simulation inside the selected portfolio with server pricing, derived holdings, and realized/unrealized P&L |
| Transaction ledger | Checking, savings, and portfolio-linked brokerage accounts; signed cash flows, filters, and manual entry |
| Classification | Editable English/German category suggestions using auditable rules plus character TF-IDF and Logistic Regression |
| Market forecasting | Versioned historical OHLCV snapshots, leakage-aware 20-day volatility targets, chronological purged splits, statistical references, Ridge evaluation, and immutable reports |
| ML lifecycle | Frozen evaluation sets, abstention metrics, feedback capture, immutable exports, candidate gates, explicit promotion, and rollback artifacts |
| Delivery | Backend/frontend tests, GitHub Actions, multi-stage images, health checks, reverse proxy, and persistent Compose storage |

Financial metrics are calculated only by deterministic backend code. A future
LLM may select tested tools and explain their output, but must never calculate
financial metrics itself. RAG and LLM integration are not implemented yet.

## Architecture

```text
Browser -> React / Nginx -> FastAPI -> SQLite + cached market data
                              |----> deterministic portfolio analytics
                              `----> rules + bilingual expense classifier

Offline ML commands -> versioned snapshots -> features -> evaluation reports
```

See the [system overview](docs/architecture/system-overview.md) for component
boundaries and data flows. Container details are in the
[container architecture](docs/architecture/containerization.md). The
[portfolio risk-score method](docs/architecture/portfolio-risk-score.md)
documents inputs, weights, thresholds, and interpretation limits.

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
.\.venv\Scripts\python.exe -m financial_ai.ml.transaction_classification.category_bootstrap
.\.venv\Scripts\python.exe -m uvicorn financial_ai.main:app --host 127.0.0.1 --port 8000 --reload
```

This fallback cannot install or update dependencies. Copy `.env.example` to
`.env` for backend overrides and `apps/web/.env.example` to `apps/web/.env` for
frontend overrides. Do not commit secrets.

Demo portfolios require no credentials and keep tests reproducible. To enable
the `External market data` option for a new paper portfolio, copy `.env.example`
to `.env` and set `FINANCIAL_AI_ALPACA_API_KEY` plus
`FINANCIAL_AI_ALPACA_SECRET_KEY`. Docker Compose reads that local file and passes
the values only to the API container; neither the browser nor the repository
contains the credentials. Search and adjusted daily bars then come from Alpaca
while orders remain simulated. A keyless SEC company catalog supports US stock
discovery without price access. Reviewers without credentials can use every demo
workflow. External data remains subject to the provider's plan, freshness,
licensing, and usage terms.

Each portfolio owns one brokerage account. Orders are priced by the backend
from the latest cached daily close and stored as regular signed ledger
transactions: a buy reduces cash and increases the holding; a sale increases
cash and reduces the holding. Portfolio cash, holdings, average cost, and P&L
are derived from the linked account and its transactions. No real brokerage
order is placed.

The Transactions view reads that same ledger. Portfolio buys and sells appear
there alongside checking and savings activity, and every account exposes a
balance derived from its opening balance plus signed cash flows. Risk analytics
replay the portfolio's current post-trade holdings before reconstructing the
historical comparison series.

## Data and intended use

The application is built as a production-oriented engineering and applied-ML
case study. Portfolio orders are simulated and no real brokerage trades are
placed. Demo portfolios, ledger activity, and transaction-classification
training data are synthetic; external portfolio prices can optionally come from
Alpaca, the volatility experiment uses a versioned historical Alpaca SIP
snapshot, and currency conversion uses a stored ECB reference snapshot.

These boundaries keep the repository reproducible while making the provenance
of real and generated observations explicit. Production use with customer data
would additionally require provider licensing, security and privacy reviews,
operational monitoring, and independent model validation. Nothing in this
repository is financial advice.

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

The next ML increment compares a nonlinear boosting model with the frozen
volatility references before any final test evaluation or product integration.
Transaction risk scoring and an assistant with deterministic tool calling remain
later product increments; cloud deployment follows after the local system and
model lifecycle are stable.
