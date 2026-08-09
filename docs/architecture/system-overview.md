# System overview

## Purpose and principles

Financial AI Assistant is a local-first portfolio project that demonstrates
backend financial analytics, an account and transaction ledger, applied text
classification, and a guarded offline ML lifecycle.

Three boundaries shape the design:

1. Financial calculations live in deterministic, tested backend services.
2. The browser displays API results and never reimplements financial logic.
3. A future LLM may orchestrate tools and explain sourced results, but may not
   calculate financial metrics itself.

## Components

```text
Browser
  |
  v
React dashboard ---- /api ----> FastAPI
                                |-- portfolio analytics
                                |-- portfolio trading and analytics
                                |-- linked accounts and transaction ledger
                                |-- classification orchestrator
                                |     |-- bilingual text rules
                                |     `-- TF-IDF + Logistic Regression
                                |-- feedback lifecycle commands
                                |
                                |-- SQLite database
                                |-- generated ML artifacts
                                `-- synthetic/external prices + stored ECB FX
```

| Component | Location | Responsibility |
|---|---|---|
| React application | `apps/web/src` | UI, API consumption, charts, editable classification suggestions |
| FastAPI application | `apps/api/src/financial_ai` | HTTP contracts, validation, persistence, analytics, and trusted classification |
| Database migrations | `apps/api/alembic` | Versioned SQLite schema changes |
| Transaction classification ML | `apps/api/src/financial_ai/ml/transaction_classification` | Taxonomy, data preparation, baselines, artifact building, evaluation, and feedback lifecycle |
| Market forecast ML | `apps/api/src/financial_ai/ml/market_forecast` | Versioned market data, leakage-aware features, temporal model selection, and offline evaluation |
| Versioned inputs | `data/market`, `data/evaluation` | ECB snapshot and frozen evaluation assets |
| Local runtime state | `data/runtime` | SQLite, generated datasets, models, exports, and reports; ignored by Git |
| Market-data service | `market_data_service.py` | Provider-neutral discovery, daily-price retrieval, persistent caching, provenance, and freshness |
| Containers | `docker`, `compose.yaml` | Reproducible API and production web runtime |

## Main data flows

### Portfolio analytics

The API derives current holdings by replaying opening positions and security
transactions from the portfolio-linked brokerage account. It then loads
price/FX observations from backend providers and calculates valuation, cost
basis, P&L, allocations, return, volatility, drawdown, and concentration. A buy
or sale therefore updates both the account ledger and the risk analytics. The
React application receives final values and time series for presentation.

A versioned deterministic risk-indicator service separates measured market
risk from diversification quality and liquidity resilience. The response
contains component inputs, weights, main drivers, and interpretation limits. See the
[risk-score method](portfolio-risk-score.md).

Demo security prices are deterministic synthetic data; external portfolios use
cached Alpaca daily observations with source and freshness metadata. ECB USD,
GBP, and JPY reference rates are stored with provenance and inverted into EUR
conversion rates. Historical risk reconstructs today's quantities backwards
and is not actual account performance.

The market-data foundation exposes instrument search, latest daily quotes, and
history through a provider-neutral service. Each portfolio permanently selects
credential-free `demo` data or `external` Alpaca observations. US company names
remain discoverable through the keyless SEC catalog. Instruments and a bounded
daily history are cached in SQLite with provider, observation, and retrieval
timestamps. External data is not silently presented as live: the UI and API
expose its source and cache freshness, while all orders remain simulated.

### Transaction categorization

The classifier receives description, optional counterparty, and signed amount.
Small bilingual rules handle high-signal phrases. Unmatched outflows use the
expense-only character TF-IDF and Logistic Regression model; unmatched inflows
abstain. The user may accept, replace, or omit the suggestion.

The backend repeats classification when saving. Browser-supplied confidence or
model metadata is therefore never trusted. Prediction, final label, method,
versions, and review state are stored for auditability.

### Portfolio trading and cash ledger

Trading is part of the selected portfolio rather than a separate portfolio
domain. Every portfolio links to one brokerage account. A user selects an
instrument discovered through that portfolio's market-data provider; the backend
obtains the latest cached daily close and records the simulated order as a
regular security transaction. The browser cannot supply an execution price.

The signed transaction amount is negative for a buy and positive for a sale.
Cash is the account opening balance plus all ledger cash flows; holdings replay
opening positions and security transactions. Foreign-currency execution values
are converted into the brokerage account currency with the stored FX reference
rate. Orders reject insufficient cash, short positions, and currencies without
an available FX rate. Client order IDs make identical retries idempotent.
Execution remains simulated: there is no broker connection, and fees, taxes,
spreads, and slippage are currently zero.

Order dates and market-data dates are separate domain values. `booked_at` is
the order's calendar date in `FINANCIAL_AI_APP_TIMEZONE`; `created_at` is the
technical UTC event timestamp; and `price_observed_on` is the date of the market
observation used for pricing. A stale daily close therefore never backdates the
ledger transaction.

The general Transactions view reads the same rows. Brokerage accounts expose
their linked portfolio identity and every account balance is derived from its
opening balance plus signed ledger entries. Market orders use the deterministic
`Investments` category because their transaction semantics are already known;
free-form cash transactions continue through the editable classification flow.

### Offline feedback lifecycle

```text
saved feedback -> immutable export -> human review -> isolated candidate
     -> fixed evaluation gates -> explicit promotion -> archived predecessor
```

There is no online retraining. Exports exclude identifiers and amount magnitude,
but free text can still be sensitive. Candidates cannot overwrite the active
model during training, and promotion requires passing machine-readable gates
plus an explicit command.

## Runtime and delivery

Local development runs Vite and FastAPI separately. Docker Compose runs an
unprivileged Nginx container for the built React application and an unprivileged
FastAPI container. Nginx proxies `/api` to FastAPI; a named volume preserves
SQLite and generated ML artifacts. GitHub Actions verifies code, builds both
images, starts the stack, and probes the browser-facing proxy path.

## Current boundaries

- The application has no authentication and must not hold real personal data.
- SQLite and the shared local runtime volume target a single-instance demo, not
  horizontal production deployment.
- Classifier training and evaluation data is synthetic; probabilities are not
  calibrated and reported metrics are development evidence only.
- No LLM, RAG, cloud resources, or secrets manager is implemented. Alpaca is an
  optional external daily-market-data provider; it is not a brokerage connection.
- Production deployment would require identity and authorization, an external
  database, durable object/model storage, privacy controls, monitoring, TLS,
  secrets management, and licensed data.

The current ML sequence completes volatility-model evaluation before any
product integration. Broader risk modeling, deterministic assistant tool
calling, observability, and cloud infrastructure remain later increments.
