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
uv sync --all-groups
uv run alembic upgrade head
uv run financial-ai-api
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
- `POST /v1/transactions/classify`

The transaction list supports account, type, category, date-range, limit, and
offset filters. Security buy and sell requests require a brokerage account,
symbol, quantity, and unit price. All bundled transactions and counterparties
are synthetic demo data.

Build the local bilingual transaction-category model before requesting ML-based
expense classification:

```powershell
uv run financial-ai-build-category-model
```

Structured types such as salary, interest, fees, and security transactions are
categorized deterministically and do not require the model artifact.
The add-transaction form requests a category suggestion after a short typing
delay, keeps the category editable, and stores the trusted backend prediction
alongside the user's final selection for later offline model improvement.

## Documentation

Detailed technical and ML documentation is maintained under `docs/`:

- [Transaction category taxonomy](docs/ml/transaction_categories.md) defines
  the versioned expense labels, scope, examples, and boundary rules.
- [Transaction classification methodology](docs/ml/transaction_classification.md)
  records dataset provenance, preprocessing decisions, assumptions, evaluation
  protocol, baseline results, limitations, and planned model development.
- [German transaction challenge set](docs/ml/german_transaction_challenge.md)
  defines the versioned synthetic German evaluation dataset, schema, review
  rules, and multilingual evaluation protocol.
- [German synthetic transaction training data](docs/ml/german_transaction_training.md)
  documents the version 1 diagnostic generator, provenance fields, leakage
  controls, reproducibility contract, and limitations.
- [German synthetic transaction training data v2](docs/ml/german_transaction_training_v2.md)
  records the harder merchant-, detail-, and format-holdout design and its
  validation results.
- [Multilingual transaction classification](docs/ml/multilingual_transaction_classification.md)
  compares English-only, German-only, and bilingual validation baselines while
  keeping all test datasets frozen.
- [Controlled English synthetic training data](docs/ml/controlled_english_training.md)
  mirrors the German v2 provenance holdouts and records which aggregate legacy
  train patterns informed its bank-description formats.
- [Transaction classifier service](docs/ml/transaction_classifier_service.md)
  documents deterministic routing, model artifact provenance, confidence and
  review behavior, API usage, configuration, and operational limitations.
- [Transaction classification feedback loop](docs/ml/transaction_classification_feedback.md)
  documents automatic editable suggestions, persisted model provenance and user
  outcomes, failure behavior, and the guarded offline-retraining protocol.

New documentation should be added to this index when it is introduced so the
repository documentation remains discoverable from the project entry point.

## Tests

```powershell
uv run pytest
cd apps/web
npm.cmd test
npm.cmd run build
```

If `uv` is not found after installation, open a new terminal so its updated
`PATH` is loaded. Use `npm.cmd` in PowerShell environments where the execution
policy blocks `npm.ps1`. Generated local content in `.venv`, `node_modules`,
`dist`, and `data/runtime` is excluded from version control.

The application is educational software, not financial advice. All included
security prices are synthetic demo data.
