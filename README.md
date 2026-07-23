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

## Tests

```powershell
uv run pytest
cd apps/web
npm test
npm run build
```

The application is educational software, not financial advice. All included
security prices are synthetic demo data.
