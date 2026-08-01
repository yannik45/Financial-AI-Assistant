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
                                |-- accounts and transaction ledger
                                |-- classification orchestrator
                                |     |-- bilingual text rules
                                |     `-- TF-IDF + Logistic Regression
                                |-- feedback lifecycle commands
                                |
                                |-- SQLite database
                                |-- generated ML artifacts
                                `-- synthetic prices + stored ECB FX snapshot
```

| Component | Location | Responsibility |
|---|---|---|
| React application | `apps/web/src` | UI, API consumption, charts, editable classification suggestions |
| FastAPI application | `apps/api/src/financial_ai` | HTTP contracts, validation, persistence, analytics, and trusted classification |
| Database migrations | `apps/api/alembic` | Versioned SQLite schema changes |
| ML modules | `apps/api/src/financial_ai/ml` | Data preparation, baselines, artifact building, evaluation, and feedback lifecycle |
| Versioned inputs | `data/market`, `data/evaluation` | ECB snapshot and frozen evaluation assets |
| Local runtime state | `data/runtime` | SQLite, generated datasets, models, exports, and reports; ignored by Git |
| Market-data service | `market_data_service.py` | Provider-neutral discovery, daily-price retrieval, persistent caching, provenance, and freshness |
| Containers | `docker`, `compose.yaml` | Reproducible API and production web runtime |

## Main data flows

### Portfolio analytics

The API loads holdings from SQLite and price/FX observations from backend data
providers. Analytics calculates valuation, cost basis, P&L, allocations, return,
volatility, drawdown, and concentration. The React application receives final
values and time series for presentation.

Security prices are deterministic synthetic data. ECB USD, GBP, and JPY
reference rates are stored with provenance and inverted at runtime to the EUR
conversion required by analytics. Historical risk reconstructs today's
quantities backwards and is not actual account performance.

The market-data foundation also exposes instrument search, latest daily quotes,
and history through a provider-neutral service. Demo mode is the credential-free
default. An optional Twelve Data adapter can retrieve external observations;
instruments and daily prices are cached in SQLite with provider, observation,
and retrieval timestamps. External data is not silently presented as live:
responses expose its source and cache freshness.

### Transaction categorization

The classifier receives description, optional counterparty, and signed amount.
Small bilingual rules handle high-signal phrases. Unmatched outflows use the
expense-only character TF-IDF and Logistic Regression model; unmatched inflows
abstain. The user may accept, replace, or omit the suggestion.

The backend repeats classification when saving. Browser-supplied confidence or
model metadata is therefore never trusted. Prediction, final label, method,
versions, and review state are stored for auditability.

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
- No LLM, RAG, cloud resources, secrets manager, or live market-data provider is
  implemented.
- Production deployment would require identity and authorization, an external
  database, durable object/model storage, privacy controls, monitoring, TLS,
  secrets management, and licensed data.

The planned sequence is fraud/risk scoring, then RAG with deterministic tool
calling, followed later by cloud infrastructure and deeper observability.
